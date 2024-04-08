# Copyright (c) 2021-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import pickle
import torch
import time

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.utils.math_utils import *
from isaacgymenvs.tasks.base.vec_task import HardwareVecTask
from isaacgymenvs.tasks.amp.poselib.poselib.core.rotation3d import *
from isaacgymenvs.utils.asset_utils import *
from isaacgymenvs.utils.franka_task_utils import create_franka_wall_envs, X_WR
from isaacgymenvs.utils.utils import BCOLORS
from scipy.spatial.transform import Rotation as R
import wandb


class FrankaYCBPush(HardwareVecTask):
    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        self.cfg = cfg
        self.rs = np.random.RandomState(43)

        # self.action_scale = self.cfg["env"]["actionScale"]
        self.object_pose_noises = self.cfg["env"]["objectPoseNoises"]

        self.franka_position_noise = (
            self.cfg["env"]["frankaPositionNoise"]
            if not self.cfg["run_cfg"]["hardware_action"]
            else 0
        )
        self.franka_rotation_noise = (
            self.cfg["env"]["frankaRotationNoise"]
            if not self.cfg["run_cfg"]["hardware_action"]
            else 0
        )

        # Always use the default dof pose if we are using hardware
        self.franka_dof_noise = (
            self.cfg["env"]["frankaDofNoise"]
            if not self.cfg["run_cfg"]["hardware_action"]
            else 0.0
        )
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # Create dicts to pass to reward function
        self.reward_settings = {
            "r_dist_scale": self.cfg["env"]["distRewardScale"],
            "r_lift_scale": self.cfg["env"]["liftRewardScale"],
            "r_align_scale": self.cfg["env"]["alignRewardScale"],
            "r_success_scale": self.cfg["env"]["successRewardScale"],
        }

        # Controller type
        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {
            "osc",
            "joint_tor",
            "joint_imp",
        }, "Invalid control type specified. Must be one of: {osc, joint_tor,joint_imp}"

        # Latent variable dimension for one-hot encoding
        # This should never change because the network input size is fixed
        self.latent_dim = len(self.cfg["env"]["chosen_objs"])

        self.obs_list = FrankaYCBPush.base_obs_list.copy()
        if self.control_type == "osc":
            self.obs_list.append(["q_gripper", 1])
        elif self.control_type == "joint_tor" or self.control_type == "joint_imp":
            self.obs_list.append(["q", 9])
        if self.cfg["task"]["use_latent"]:
            self.obs_list.append(["cubeA_latent", self.latent_dim])
        # drop the number in obs_list
        self.obs_names = [self.obs_list[i][0] for i in range(len(self.obs_list))]
        # Dictionary specifying the index trange each observation corresponds to
        self.obs_dict = {}
        running_sum = 0
        for i in range(len(self.obs_list)):
            self.obs_dict[self.obs_list[i][0]] = [
                running_sum,
                running_sum + self.obs_list[i][1],
            ]
            running_sum += self.obs_list[i][1]
        # dimensions
        self.cfg["env"]["numObservations"] = running_sum
        # actions include: delta EEF if OSC (6) or joint torques (7) + bool gripper (1)
        self.cfg["env"]["numActions"] = 7 if self.control_type == "osc" else 8
        # Values to be filled in at runtime
        self.states = (
            {}
        )  # will be dict filled with relevant states to use for reward calculation
        self.handles = {}  # will be dict mapping names to relevant sim handles
        self.num_dofs = None  # Total number of DOFs per env
        self._init_cubeA_state = None  # Initial state of cubeA for the current env
        self._init_cubeB_state = None  # Initial state of cubeB for the current env
        self._cubeA_state = None  # Current state of cubeA for the current env
        self._cubeB_state = None  # Current state of cubeB for the current env
        self._cubeA_id = None  # Actor ID corresponding to cubeA for a given env
        self._cubeB_id = None  # Actor ID corresponding to cubeB for a given env
        self._init_franka_state = None

        # Tensor placeholders
        self._root_state = None  # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None  # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = (
            None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        )
        self._contact_forces = None  # Contact forces in sim
        self._eef_state = None  # end effector state (at grasping point)
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._arm_control = None  # Tensor buffer for controlling arm
        self._gripper_control = None  # Tensor buffer for controlling gripper
        self._pos_control = None  # Position actions
        self._effort_control = None  # Torque actions
        self._franka_effort_limits = None  # Actuator effort limits for franka
        self._global_indices = (
            None  # Unique indices corresponding to all envs in flattened array
        )

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2
        # Mask out the latent dimesions
        self.observation_dr_mask = torch.ones(
            self.cfg["env"]["numObservations"], device=rl_device
        )
        self.observation_dr_mask[-self.latent_dim :] = 0
        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        # Read off primitive sequence if there is one. Note that this is never binding
        # because the time limit of each primitive will trigger before the total time.

        # OSC Gains
        self.kp = to_torch([300.0, 300, 300, 30, 30, 30], device=self.device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = to_torch([10.0] * 7, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)
        # self.cmd_limit = None                   # filled in later

        self.cfg["env"]["stateStartEndFile"] = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            self.cfg["env"]["stateStartEndFile"],
        )
        # load the initial state
        with open(self.cfg["env"]["stateStartEndFile"], "rb") as pickle_file:
            self.initial_states = pickle.load(pickle_file)

        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs(
            self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs))
        )
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        create_franka_wall_envs(self, num_envs, spacing, num_per_row)
        # Setup init state buffer
        self._init_cubeA_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_cubeB_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_franka_state = torch.zeros(self.num_envs, 9, device=self.device)
        # Setup data
        self.init_data()

    def init_latent(self):
        # Initialize latent states using self.cfg['env']['chosen_objs']
        # Note that this could be different from self.chosen_objs if the user specifies a different object via self.cfg["run_cfg"]["hardware_object"]
        latents = torch.zeros(self.num_envs, self.latent_dim, device=self.device)  # nx2
        if self.cfg["run_cfg"]["hardware_object"] is not None:
            # find the index of the object in the chosen_objs list
            obj_idx = self.cfg["env"]["chosen_objs"].index(
                self.cfg["run_cfg"]["hardware_object"]
            )
            latents[:, obj_idx] = 1
        else:
            # Initialize latent states
            for i in range(self.latent_dim):
                latents[i :: self.latent_dim, i] = 1
        self.states.update(
            {
                "cubeA_latent": latents,
                "cubeB_latent": latents,
            }
        )

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        franka_handle = 0
        self.handles = {
            # Franka
            "hand": self.gym.find_actor_rigid_body_handle(
                env_ptr, franka_handle, "panda_hand"
            ),
            "leftfinger_tip": self.gym.find_actor_rigid_body_handle(
                env_ptr, franka_handle, "panda_leftfinger_tip"
            ),
            "rightfinger_tip": self.gym.find_actor_rigid_body_handle(
                env_ptr, franka_handle, "panda_rightfinger_tip"
            ),
            "grip_site": self.gym.find_actor_rigid_body_handle(
                env_ptr, franka_handle, "panda_grip_site"
            ),
            # Cubes
            "cubeA_body_handle": self.gym.find_actor_rigid_body_handle(
                self.envs[0], self._cubeA_id, "rigid_body"
            ),
            "cubeB_body_handle": self.gym.find_actor_rigid_body_handle(
                self.envs[0], self._cubeB_id, "rigid_body"
            ),
        }
        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(
            self.num_envs, -1, 13
        )
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(
            self.num_envs, -1, 2
        )
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(
            self.num_envs, -1, 13
        )
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._eef_state = self._rigid_body_state[:, self.handles["grip_site"], :]
        self._eef_lf_state = self._rigid_body_state[
            :, self.handles["leftfinger_tip"], :
        ]
        self._eef_rf_state = self._rigid_body_state[
            :, self.handles["rightfinger_tip"], :
        ]
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        hand_joint_index = self.gym.get_actor_joint_dict(env_ptr, franka_handle)[
            "panda_hand_joint"
        ]
        self._j_eef = jacobian[:, hand_joint_index, :, :7]
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :7, :7]
        self._cubeA_state = self._root_state[:, self._cubeA_id, :]
        self._cubeB_state = self._root_state[:, self._cubeB_id, :]
        # Initialize states
        self.states.update(
            {
                "cubeA_size": torch.ones_like(self._eef_state[:, 0]) * self.cubeA_size,
                "cubeB_size": torch.ones_like(self._eef_state[:, 0]) * self.cubeB_size,
            }
        )
        if self.cfg["task"]["use_latent"]:
            self.init_latent()

        # Initialize actions
        self._pos_control = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize control
        self._arm_control = self._effort_control[:, :7]
        self._gripper_control = self._pos_control[:, 7:9]

        # Initialize indices
        # Important: this should be modified if extra assets are added
        self._global_indices = torch.arange(
            self.num_envs * 6, dtype=torch.int32, device=self.device
        ).view(self.num_envs, -1)

    def _update_states(self):
        self.states.update(
            {
                # Franka
                "q": self._q[:, :],
                "q_gripper": self._q[:, -2:],
                "eef_pos": self._eef_state[:, :3],
                "eef_quat": self._eef_state[:, 3:7],
                "eef_vel": self._eef_state[:, 7:],
                "eef_lf_pos": self._eef_lf_state[:, :3],
                "eef_rf_pos": self._eef_rf_state[:, :3],
                # Cubes
                "cubeA_quat": self._cubeA_state[:, 3:7],
                "cubeA_pos": self._cubeA_state[:, :3],
                "cubeA_pos_relative": self._cubeA_state[:, :3] - self._eef_state[:, :3],
                "cubeA_to_cubeB_rot_rad": quat_diff_rad(
                    self._cubeA_state[:, 3:7], self._cubeB_state[:, 3:7]
                ).reshape(-1, 1),
                "cubeB_quat": self._cubeB_state[:, 3:7],
                "cubeB_pos": self._cubeB_state[:, :3],
                "cubeA_to_cubeB_pos": self._cubeB_state[:, :3]
                - self._cubeA_state[:, :3],
                "q_target": self._q_target[
                    :, :8
                ],  # note that last 2 entries are identical
                # # Latents do not need to be updated
                # "cubeA_latent": torch.randn(
                #     self.num_envs, self.latent_dim, device=self.device
                # ),
                # "cubeB_latent": torch.randn(
                #     self.num_envs, self.latent_dim, device=self.device
                # ),
            }
        )
        # Override with robot hardware readings if applicable
        if self.has_hardware_action:
            self.states["q"][:, :7] = torch.tensor(
                self.robot.last_arm_q, device=self.device
            )
            self.states["q"][:, 7:] = torch.tensor(
                self.robot.last_gripper_q, device=self.device
            )
            self.states["q_gripper"][:] = torch.tensor(
                self.robot.last_gripper_q, device=self.device
            )
            robot_eef_pose = X_WR @ self.robot.last_eef_pose
            self.states["eef_pos"][:] = torch.tensor(
                robot_eef_pose[:3, -1], device=self.device
            )
            robot_eef_state = X_to_state(robot_eef_pose)
            self.states["eef_quat"][:] = torch.tensor(
                robot_eef_state[3:7], device=self.device
            )
            self.states["eef_pos"][:] = torch.tensor(
                robot_eef_state[:3], device=self.device
            )

        # TODO: override object state too?

        # Save the observation
        if self.save_states:
            # Note that saving only works with 1 env for now
            timestep = self.progress_buf.detach().cpu().numpy()[0]
            if timestep not in self.state_history:
                self.state_history[timestep] = {}
            for key in self.states.keys():
                self.state_history[timestep][key] = (
                    self.states[key].detach().cpu().numpy()
                )

    def _refresh_sim_tensors(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Refresh states

    def compute_reward(self):
        cubeA_align_cubeB = (
            torch.norm(self.states["cubeA_to_cubeB_pos"][:, :2], dim=-1) < 0.02
        )
        self.rew_buf[:], self.reset_buf[:], self.success_buf[:] = compute_franka_reward(
            self.reset_buf,
            self.progress_buf,
            self.success_buf,
            self.states,
            self.reward_settings,
            self.max_episode_length,
            self.has_hardware,
        )

    def compute_observations(self):
        self._update_states()
        # Note that obs_buf goes through DR eventually, but the state doesn't
        self.obs_buf = torch.cat([self.states[ob] for ob in self.obs_names], dim=-1)
        # breakpoint()  # check what obs_buf is
        return self.obs_buf

    def reset_idx(self, env_ids):
        # TODO: can we reset to different objects?
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        if self.has_hardware:
            if self.has_hardware_action:
                self.reset_hardware_robot()
                self._pos_control[env_ids, :] = self._q[:].clone()
            input("Reset object and press enter to continue...")

        self._reset_init_cube_state_joint(env_ids)
        # Reset the internal obs accordingly
        if not self.has_hardware_action:
            # Reset agent
            reset_noise = torch.rand((len(env_ids), 9), device=self.device)
            pos = tensor_clamp(
                self._init_franka_state[env_ids]  # .unsqueeze(0)
                + self.franka_dof_noise * 2.0 * (reset_noise - 0.5),
                self.franka_dof_lower_limits.unsqueeze(0),
                self.franka_dof_upper_limits.unsqueeze(0),
            )
            # Overwrite gripper init pos (no noise since these are always position controlled)
            pos[:, -2:] = self._init_franka_state[env_ids, -2:]
            self._q[env_ids, :] = pos
            self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])
            # Reset the impedance control
            self._q_target[env_ids, :] = self._q[env_ids, :]
            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self._dof_state),
                gymtorch.unwrap_tensor(multi_env_ids_int32),
                len(multi_env_ids_int32),
            )
            self._pos_control[env_ids, :] = pos

        if self.raw_primitive_sequence is not None:
            # Update the primitives such that it starts with the current observation
            self.retarget_primitives()

        # Reset the cube after resetting the robot to avoid occlusions when running on hardware
        # Write these new init states to the sim states
        self._cubeA_state[env_ids] = self._init_cubeA_state[env_ids]
        self._cubeB_state[env_ids] = self._init_cubeB_state[env_ids]

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._effort_control[env_ids, :] = 0.0

        # Deploy updates
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._pos_control),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )
        self.gym.set_dof_actuation_force_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._effort_control),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )
        # Update cube states
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, -2:].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32),
            len(multi_env_ids_cubes_int32),
        )
        if self.save_states and self.progress_buf[0] > 0:
            # Ignore the first reset, which is before any simulation occurs
            self.save_and_clear_state_history()
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.success_buf[env_ids] = 0
        self._refresh_sim_tensors()

    def _reset_init_cube_state_joint(self, env_ids):
        """
        Simple method to sample @cube's position based on self.startPositionNoise and self.startRotationNoise, and
        automaticlly reset the pose internally. Populates the appropriate self._init_cubeX_state

        If @check_valid is True, then this will also make sure that the sampled position is not in contact with the
        other cube.

        Args:
            cube(str): Which cube to sample location for. Either 'A' or 'B'
            env_ids (tensor or None): Specific environments to reset cube for
            check_valid (bool): Whether to make sure sampled position is collision-free with the other cube.
        """
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(
                start=0, end=self.num_envs, device=self.device, dtype=torch.long
            )
        num_resets = len(env_ids)
        obj_idx_at_env = (env_ids % len(self.chosen_objs)).cpu().numpy()
        # Load the object B (target pose) state from the dataset
        sampled_cubeA_state = torch.zeros(num_resets, 13, device=self.device)
        sampled_cubeB_state = torch.zeros(num_resets, 13, device=self.device)
        sampled_franka_state = torch.zeros(num_resets, 9, device=self.device)
        for reset_idx in range(num_resets):
            obj_idx = int(obj_idx_at_env[reset_idx])
            obj_str = self.chosen_objs[obj_idx]
            state_idx = self.rs.choice(len(self.initial_states[obj_str]))
            # TODO

            assert self.initial_states[obj_str][state_idx]["obj_name"] == obj_str

            selected_state_data = self.initial_states[obj_str][state_idx]
            initial_state = torch.tensor(
                selected_state_data["initial_state"], device=self.device
            )
            final_state = torch.tensor(
                selected_state_data["final_state"], device=self.device
            )
            sampled_cubeA_state[reset_idx, :7] = initial_state

            sampled_cubeB_state[reset_idx, :7] = final_state
            # breakpoint()
            # interpolate start state
            interpolated_pos = (
                self.rs.rand() * (final_state - initial_state) + initial_state
            )[:3]
            sampled_cubeA_state[reset_idx, :3] = interpolated_pos
            if "init_franka_q" in selected_state_data:
                sampled_franka_state[reset_idx, :] = torch.tensor(
                    selected_state_data["init_franka_q"], device=self.device
                )
            else:
                sampled_franka_state[reset_idx, :] = torch.tensor(
                    self.franka_default_dof_pos, device=self.device
                )

            # wall_state = self.initial_states[ycb_id][state_idx][0]
            # Cube B should be the state prior to pivoting (i.e. wall_state)
            # rand_yaw = np.pi / 2.0 * np.random.randint(0, 4)
            # rand_roll = np.pi * np.random.randint(0, 2)
            # pitch = np.pi / 2.0  # * np.random.choice([-1, 1])
            # rand_quat_cubeB = torch.tensor(
            #     R.from_euler(
            #         "zyz",  # roll pitch yaw
            #         [
            #             rand_roll,
            #             pitch,
            #             rand_yaw,
            #             # + np.random.uniform(-1, 1.0)
            #             # * self.object_pose_noises["startRotationNoise"],
            #         ],
            #     ).as_quat()
            # ).to(self.device)
            # sampled_cubeB_state[reset_idx, 3:7] = rand_quat_cubeB
            # Set the cubeB position to be close to the wall
            # sampled_cubeB_state[reset_idx, 0] = 0.18 + np.random.uniform(-0.02, 0.02)
            # sampled_cubeB_state[reset_idx, 1] = np.random.uniform(-0.25, 0.25)

            # sampled_cubeB_state[reset_idx, :7] = torch.tensor(
            #     wall_state,
            #     device=self.device,
            # )
            # sampled_cubeA_state[reset_idx, :7] = torch.clone(
            #     sampled_cubeB_state[reset_idx, :7]
            # )
            # Add in the table height as it is normalized
            # sampled_cubeA_state[reset_idx, 2] += self._table_surface_pos[2]
            # sampled_cubeB_state[reset_idx, 2] += self._table_surface_pos[2]
            # Apply random rotation to cubeA state
            rand_quat_cubeA = torch.tensor(
                R.from_euler(
                    "xyz",
                    [
                        0.0,
                        0.0,
                        np.random.uniform(-1, 1.0)
                        * self.object_pose_noises["startRotationNoise"],
                    ],
                ).as_quat()
            ).to(self.device)
            sampled_cubeA_state[reset_idx, 3:7] = quat_mul(
                rand_quat_cubeA, sampled_cubeA_state[reset_idx, 3:7]
            )

            if self.is_test and self.retargeted_primitive_sequence is not None:
                # Override the initial & goal states if specified by the primitive
                if self.retargeted_primitive_sequence[0].start_pose_WB is not None:
                    # has primitives, set cubeB to the first primitive's goal_state
                    sampled_cubeA_state[reset_idx, :7] = torch.tensor(
                        self.retargeted_primitive_sequence[0].start_pose_WB,
                        device=self.device,
                    )
                if self.retargeted_primitive_sequence[0].goal_pose_WB is not None:
                    sampled_cubeB_state[reset_idx, :7] = torch.tensor(
                        self.retargeted_primitive_sequence[0].goal_pose_WB,
                        device=self.device,
                    )
        # Set the random positions.
        target_position_noise = torch.tensor(
            self.object_pose_noises["targetPositionNoise"]
        )
        cubeB_offsets = sample_uniform_random_translations(
            -target_position_noise,
            target_position_noise,  # table is 0.6x1.0
            num_samples=num_resets,
        ).to(self.device)
        # start_rel_position_min = torch.tensor(
        #     self.object_pose_noises["startRelPositionMin"]
        # )
        # start_rel_position_max = torch.tensor(
        #     self.object_pose_noises["startRelPositionMax"]
        # )
        # cubeA_rel_offsets = sample_uniform_random_translations(
        #     start_rel_position_min,
        #     start_rel_position_max,  # table is 0.6x1.0
        #     num_samples=num_resets,
        # ).to(self.device)
        # sampled_cubeA_state[:, 1] = cubeB_offsets[:,1] #+ cubeA_rel_offsets
        # sampled_cubeB_state[:, :3] = cubeB_offsets
        if self.simulate_from_hardware_initial_state or self.has_hardware_action:
            # raise NotImplementedError
            # Reset the initial state to the object state
            X_WB_guess = state_to_X(
                sampled_cubeB_state[reset_idx, :7].detach().cpu().numpy()
            )
            # only set the orientation initial guess to avoid ambiguity from symmetry
            X_WB = self.get_hardware_object_X_WB(
                R_WB_guess=X_WB_guess[:3, :3], filter=False
            )
            print("Initial X_WB", X_WB)
            hardware_cubeA_quat = R.from_matrix(X_WB[:3, :3]).as_quat()
            sampled_cubeA_state[reset_idx, :2] = torch.tensor(
                X_WB[:2, 3],
                device=self.device,
            )
            sampled_cubeA_state[reset_idx, 3:7] = torch.tensor(
                hardware_cubeA_quat,
                device=self.device,
            )
            # Match the Y coord
            sampled_cubeB_state[reset_idx, 1] = sampled_cubeA_state[
                reset_idx, 1
            ].clone()
            # Hack to prevent the object from flying off the table due to
            # interpenetration in initial state
            sampled_cubeB_state[reset_idx, 2] += 0.03
            # Match the orientation
            sampled_cubeB_state[reset_idx, 3:7] = torch.tensor(hardware_cubeA_quat)
            # Turn off the hardware observation if doesn't have hardware action
            self.has_hardware_observation = self.has_hardware_action

        # Lastly, set these sampled values as the new init state
        self._init_cubeA_state[env_ids, :] = sampled_cubeA_state
        self._init_cubeB_state[env_ids, :] = sampled_cubeB_state
        self._init_franka_state[env_ids, :] = sampled_franka_state

    def post_physics_step(self):
        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # Grab relevant states to visualize
            eef_pos = self.states["eef_pos"]
            eef_rot = self.states["eef_quat"]
            cubeA_pos = self.states["cubeA_pos"]
            cubeA_rot = self.states["cubeA_quat"]
            cubeB_pos = self.states["cubeB_pos"]
            cubeB_rot = self.states["cubeB_quat"]
            # Plot visualizations
            for i in range(self.num_envs):
                for pos, rot in zip(
                    (eef_pos, cubeA_pos, cubeB_pos), (eef_rot, cubeA_rot, cubeB_rot)
                ):
                    px = (
                        (
                            pos[i]
                            + quat_apply(
                                rot[i], to_torch([1, 0, 0], device=self.device) * 0.2
                            )
                        )
                        .cpu()
                        .numpy()
                    )
                    py = (
                        (
                            pos[i]
                            + quat_apply(
                                rot[i], to_torch([0, 1, 0], device=self.device) * 0.2
                            )
                        )
                        .cpu()
                        .numpy()
                    )
                    pz = (
                        (
                            pos[i]
                            + quat_apply(
                                rot[i], to_torch([0, 0, 1], device=self.device) * 0.2
                            )
                        )
                        .cpu()
                        .numpy()
                    )

                    p0 = pos[i].cpu().numpy()
                    self.gym.add_lines(
                        self.viewer,
                        self.envs[i],
                        1,
                        [p0[0], p0[1], p0[2], px[0], px[1], px[2]],
                        [0.85, 0.1, 0.1],
                    )
                    self.gym.add_lines(
                        self.viewer,
                        self.envs[i],
                        1,
                        [p0[0], p0[1], p0[2], py[0], py[1], py[2]],
                        [0.1, 0.85, 0.1],
                    )
                    self.gym.add_lines(
                        self.viewer,
                        self.envs[i],
                        1,
                        [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]],
                        [0.1, 0.1, 0.85],
                    )


#####################################################################
###=========================jit functions=========================###
#####################################################################


# @torch.jit.script
def compute_franka_reward(
    reset_buf,
    progress_buf,
    success_buf,
    states,
    reward_settings,
    max_episode_length,
    has_hardware,
):
    # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor]

    # Compute per-env physical parameters
    # target_height = states["cubeB_size"] + states["cubeA_size"] / 2.0
    # cubeA_size = states["cubeA_size"]
    # cubeB_size = states["cubeB_size"]

    # distance from hand to the cubeA
    d = torch.norm(states["cubeA_pos_relative"], dim=-1)
    d_lf = torch.norm(states["cubeA_pos"] - states["eef_lf_pos"], dim=-1)
    d_rf = torch.norm(states["cubeA_pos"] - states["eef_rf_pos"], dim=-1)
    # dist_reward = 1 - torch.tanh(10.0 * (d + d_lf + d_rf) / 3)
    dist_reward = 1.0 - torch.tanh(
        10.0 * torch.clamp((d + d_lf + d_rf) / 3 - 0.16, 0.0)
    )

    # Rotation error
    cubeA_to_cubeB_rot_rad = states["cubeA_to_cubeB_rot_rad"]

    # how closely aligned cubeA is to cubeB
    d_ab = torch.norm(states["cubeA_to_cubeB_pos"], dim=-1)
    # align_reward = 1 - torch.tanh(10.0 * d_ab)
    align_reward = 1 - torch.tanh(d_ab)

    # Dist reward is maximum of dist and align reward
    # i.e. if cubeA and cubeB are aligned, then dist reward is independent of the
    # hand-object distance
    # dist_reward = torch.max(dist_reward, align_reward)

    # final reward for aligning successfully
    cubeA_align_cubeB = torch.logical_and(
        torch.norm(states["cubeA_to_cubeB_pos"][:, :2], dim=-1)
        < (0.04 if has_hardware else 0.02),
        torch.abs(cubeA_to_cubeB_rot_rad).squeeze() < (0.5 if has_hardware else 0.25),
    )
    # gripper_away_from_cubeA = d > 0.04
    success_reward = cubeA_align_cubeB  # & gripper_away_from_cubeA

    # Compose rewards

    # We either provide the stack reward or the align + dist reward
    rewards = torch.where(
        success_reward,
        reward_settings["r_success_scale"] * success_reward,  # task success
        reward_settings["r_dist_scale"] * dist_reward  # hand-obj distance
        # + reward_settings["r_lift_scale"] * lift_reward  # lift object penalty
        + reward_settings["r_align_scale"] * align_reward,  # align object reward
    )
    # Compute resets
    success_buf = torch.where(
        torch.logical_and(
            success_reward > 0,
            progress_buf > 1,  # FIXME: progress_buf > 1 shouldn't be necessary
        ),
        torch.ones_like(reset_buf),
        success_buf,
    )
    reset_buf = torch.where(
        (progress_buf >= max_episode_length - 1) | (success_buf > 0),
        torch.ones_like(reset_buf),
        reset_buf,
    )
    if wandb.run is not None:
        wandb.log(
            {
                "success_reward": torch.mean(success_reward.float()),
                "dist_reward": torch.mean(dist_reward),
                "align_reward": torch.mean(align_reward),
            }
        )
    return rewards, reset_buf, success_buf
