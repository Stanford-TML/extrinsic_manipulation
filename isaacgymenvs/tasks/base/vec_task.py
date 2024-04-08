# Copyright (c) 2018-2022, NVIDIA Corporation
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

from typing import Dict, Any, Tuple

import gym
from gym import spaces

from isaacgym import gymtorch, gymapi
import isaacgymenvs.primitives.primitive_controllers as primitive_controllers
from isaacgym.torch_utils import to_torch
from isaacgymenvs.utils.dr_utils import (
    get_property_setter_map,
    get_property_getter_map,
    get_default_setter_args,
    apply_random_samples,
    check_buckets,
    generate_random_samples,
)
from isaacgym.torch_utils import *
from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.utils.math_utils import *
from isaacgymenvs.utils.franka_task_utils import X_WR
from isaacgymenvs.utils.yaml_config import YamlConfig
from isaacgymenvs.utils.asset_utils import get_tml_asset_path
import isaacgymenvs.primitives.primitive_types as primitive_types
import isaacgymenvs.utils.deoxys_utils as deoxys_utils
from isaacgymenvs.utils.utils import BCOLORS


try:
    from contactdemo.lib.drake.contact_retargeting import (
        retarget_primitive_sequence,
        update_primitive_given_start_WB_change,
    )
    from contactdemo.lib.utils.hardware_utils import update_cfg_wall_pos_orn
except ImportError:
    print("contactdemo not installed")
    pass
import torch
import time
import numpy as np
import operator, random
from copy import deepcopy
import sys
from scipy.spatial.transform import Rotation as R
import os
import time
import pickle
import abc
from abc import ABC
import yaml
import os

EXISTING_SIM = None
SCREEN_CAPTURE_RESOLUTION = (1027, 768)


def _create_sim_once(gym, *args, **kwargs):
    global EXISTING_SIM
    if EXISTING_SIM is not None:
        return EXISTING_SIM
    else:
        EXISTING_SIM = gym.create_sim(*args, **kwargs)
        return EXISTING_SIM


class Env(ABC):
    def __init__(
        self,
        config: Dict[str, Any],
        rl_device: str,
        sim_device: str,
        graphics_device_id: int,
        headless: bool,
    ):
        """Initialise the env.

        Args:
            config: the configuration dictionary.
            sim_device: the device to simulate physics on. eg. 'cuda:0' or 'cpu'
            graphics_device_id: the device ID to render with.
            headless: Set to False to disable viewer rendering.
        """

        split_device = sim_device.split(":")
        self.device_type = split_device[0]
        self.device_id = int(split_device[1]) if len(split_device) > 1 else 0

        self.device = "cpu"
        if config["sim"]["use_gpu_pipeline"]:
            if self.device_type.lower() == "cuda" or self.device_type.lower() == "gpu":
                self.device = "cuda" + ":" + str(self.device_id)
            else:
                print(
                    "GPU Pipeline can only be used with GPU simulation. Forcing CPU Pipeline."
                )
                config["sim"]["use_gpu_pipeline"] = False

        self.rl_device = rl_device

        # Rendering
        # if training in a headless mode
        self.headless = headless

        enable_camera_sensors = config.get("enableCameraSensors", False)
        self.graphics_device_id = graphics_device_id
        if enable_camera_sensors == False and self.headless == True:
            self.graphics_device_id = -1

        self.num_environments = config["env"]["numEnvs"]
        self.num_agents = config["env"].get(
            "numAgents", 1
        )  # used for multi-agent environments
        self.num_observations = config["env"]["numObservations"]
        self.num_states = config["env"].get("numStates", 0)
        self.num_actions = config["env"]["numActions"]

        self.control_freq_inv = config["env"].get("controlFrequencyInv", 5)

        self.obs_space = spaces.Box(
            np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf
        )
        self.state_space = spaces.Box(
            np.ones(self.num_states) * -np.Inf, np.ones(self.num_states) * np.Inf
        )

        self.act_space = spaces.Box(
            np.ones(self.num_actions) * -1.0, np.ones(self.num_actions) * 1.0
        )

        self.clip_obs = config["env"].get("clipObservations", np.Inf)
        self.clip_actions = config["env"].get("clipActions", np.Inf)
        self.policy_selection_idx = config.get("policy_selection_idx", None)

    @abc.abstractmethod
    def allocate_buffers(self):
        """Create torch buffers for observations, rewards, actions dones and any additional data."""

    @abc.abstractmethod
    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """

    @abc.abstractmethod
    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset the environment.
        Returns:
            Observation dictionary
        """

    @abc.abstractmethod
    def reset_idx(self, env_ids: torch.Tensor):
        """Reset environments having the provided indices.
        Args:
            env_ids: environments to reset
        """

    @property
    def observation_space(self) -> gym.Space:
        """Get the environment's observation space."""
        return self.obs_space

    @property
    def action_space(self) -> gym.Space:
        """Get the environment's action space."""
        return self.act_space

    @property
    def num_envs(self) -> int:
        """Get the number of environments."""
        return self.num_environments

    @property
    def num_acts(self) -> int:
        """Get the number of actions in the environment."""
        return self.num_actions

    @property
    def num_obs(self) -> int:
        """Get the number of observations in the environment."""
        return self.num_observations


class VecTask(Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 24}

    def __init__(
        self,
        config,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture: bool = False,
        force_render: bool = False,
    ):
        """Initialise the `VecTask`.

        Args:
            config: config dictionary for the environment.
            sim_device: the device to simulate physics on. eg. 'cuda:0' or 'cpu'
            graphics_device_id: the device ID to render with.
            headless: Set to False to disable viewer rendering.
            virtual_screen_capture: Set to True to allow the users get captured screen in RGB array via `env.render(mode='rgb_array')`.
            force_render: Set to True to always force rendering in the steps (if the `control_freq_inv` is greater than 1 we suggest stting this arg to True)
        """
        super().__init__(config, rl_device, sim_device, graphics_device_id, headless)
        self.virtual_screen_capture = virtual_screen_capture
        self.virtual_display = None
        if self.virtual_screen_capture:
            from pyvirtualdisplay.smartdisplay import SmartDisplay

            self.virtual_display = SmartDisplay(size=SCREEN_CAPTURE_RESOLUTION)
            self.virtual_display.start()
        self.force_render = force_render

        self.sim_params = self.__parse_sim_params(
            self.cfg["physics_engine"], self.cfg["sim"]
        )
        if self.cfg["physics_engine"] == "physx":
            self.physics_engine = gymapi.SIM_PHYSX
        elif self.cfg["physics_engine"] == "flex":
            self.physics_engine = gymapi.SIM_FLEX
        else:
            msg = f"Invalid physics engine backend: {self.cfg['physics_engine']}"
            raise ValueError(msg)

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.gym = gymapi.acquire_gym()

        self.first_randomization = True
        if not hasattr(self, "observation_dr_mask"):
            self.observation_dr_mask = None
        self.original_props = {}
        self.dr_randomizations = {}
        self.actor_params_generator = None
        self.extern_actor_params = {}
        self.last_step = -1
        self.last_rand_step = -1
        for env_id in range(self.num_envs):
            self.extern_actor_params[env_id] = None
        # create envs, sim and viewer
        self.sim_initialized = False
        self.create_sim()
        self.gym.prepare_sim(self.sim)
        self.sim_initialized = True

        self.set_viewer()
        self.allocate_buffers()
        self.obs_dict = {}

    def set_viewer(self):
        """Create the viewer."""

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_S, "skip_target"
            )
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_P, "pause")
            # set the camera position based on up axis
            sim_params = self.gym.get_sim_params(self.sim)
            if sim_params.up_axis == gymapi.UP_AXIS_Z:
                cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
                cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
            else:
                cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
                cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

            cam_pos = gymapi.Vec3(-1.0, -1.5, 2.5)
            cam_target = gymapi.Vec3(0, 0, 1)

            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def allocate_buffers(self):
        """Allocate the observation, states, etc. buffers.

        These are what is used to set observations and states in the environment classes which
        inherit from this one, and are read in `step` and other related functions.

        """

        # allocate buffers
        self.obs_buf = torch.zeros(
            (
                self.num_envs,
                self.num_obs + 1 * (self.policy_selection_idx is not None),
            ),
            device=self.device,
            dtype=torch.float,
        )
        self.states_buf = torch.zeros(
            (self.num_envs, self.num_states), device=self.device, dtype=torch.float
        )
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.timeout_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.randomize_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.success_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.extras = {}

    def create_sim(
        self,
        compute_device: int,
        graphics_device: int,
        physics_engine,
        sim_params: gymapi.SimParams,
    ):
        """Create an Isaac Gym sim object.

        Args:
            compute_device: ID of compute device to use.
            graphics_device: ID of graphics device to use.
            physics_engine: physics engine to use (`gymapi.SIM_PHYSX` or `gymapi.SIM_FLEX`)
            sim_params: sim params to use.
        Returns:
            the Isaac Gym sim object.
        """
        sim = _create_sim_once(
            self.gym, compute_device, graphics_device, physics_engine, sim_params
        )
        if sim is None:
            print("*** Failed to create sim")
            quit()

        return sim

    def get_state(self):
        """Returns the state buffer of the environment (the privileged observations for asymmetric training)."""
        return torch.clamp(self.states_buf, -self.clip_obs, self.clip_obs).to(
            self.rl_device
        )

    @abc.abstractmethod
    def pre_physics_step(self, actions: torch.Tensor):
        """Apply the actions to the environment (eg by setting torques, position targets).

        Args:
            actions: the actions to apply
        """

    @abc.abstractmethod
    def post_physics_step(self):
        """Compute reward and observations, reset any environments that require it."""

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """

        # randomize actions
        if self.dr_randomizations.get("actions", None):
            actions = self.dr_randomizations["actions"]["noise_lambda"](actions)

        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # apply actions
        self.pre_physics_step(action_tensor)

        # step physics and render each frame
        for i in range(self.control_freq_inv):
            if self.force_render:
                self.render()
            self.gym.simulate(self.sim)

        # to fix!
        if self.device == "cpu":
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 1. Timeout == 1 makes sense only if the reset buffer is 1.
        self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (
            self.reset_buf != 0
        )

        # randomize observations
        if self.dr_randomizations.get("observations", None):
            self.obs_buf = self.dr_randomizations["observations"]["noise_lambda"](
                self.obs_buf
            )

        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)

        self.obs_dict["obs"] = torch.clamp(
            self.obs_buf, -self.clip_obs, self.clip_obs
        ).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return (
            self.obs_dict,
            self.rew_buf.to(self.rl_device),
            self.reset_buf.to(self.rl_device),
            self.extras,
        )

    def zero_actions(self) -> torch.Tensor:
        """Returns a buffer with zero actions.

        Returns:
            A buffer of zero torch actions
        """
        actions = torch.zeros(
            [self.num_envs, self.num_actions],
            dtype=torch.float32,
            device=self.rl_device,
        )

        return actions

    def reset_idx(self, env_idx):
        """Reset environment with indces in env_idx.
        Should be implemented in an environment class inherited from VecTask.
        """
        pass

    def reset(self):
        """Is called only once when environment starts to provide the first observations.
        Doesn't calculate observations. Actual reset and observation calculation need to be implemented by user.
        Returns:
            Observation dictionary
        """
        self.obs_dict["obs"] = torch.clamp(
            self.obs_buf, -self.clip_obs, self.clip_obs
        ).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict

    def reset_done(self):
        """Reset the environment.
        Returns:
            Observation dictionary, indices of environments being reset
        """
        done_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(done_env_ids) > 0:
            self.reset_idx(done_env_ids)

        self.obs_dict["obs"] = torch.clamp(
            self.obs_buf, -self.clip_obs, self.clip_obs
        ).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict, done_env_ids

    def render(self, mode="rgb_array"):
        """Draw the frame to the viewer, and check for keyboard events."""
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                print("action", evt.action, "val", evt.value)
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "skip_target" and evt.value > 0:
                    print("Skipping target")
                    self.reset_buf[:] = 1
                elif evt.action == "pause" and evt.value > 0:
                    print("Pausing")
                    breakpoint()
            # fetch results
            if self.device != "cpu":
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)

                # Wait for dt to elapse in real time.
                # This synchronizes the physics simulation with the rendering rate.
                self.gym.sync_frame_time(self.sim)

            else:
                self.gym.poll_viewer_events(self.viewer)

            if self.virtual_display and mode == "rgb_array":
                img = self.virtual_display.grab()
                return np.array(img)

    def __parse_sim_params(
        self, physics_engine: str, config_sim: Dict[str, Any]
    ) -> gymapi.SimParams:
        """Parse the config dictionary for physics stepping settings.

        Args:
            physics_engine: which physics engine to use. "physx" or "flex"
            config_sim: dict of sim configuration parameters
        Returns
            IsaacGym SimParams object with updated settings.
        """
        sim_params = gymapi.SimParams()

        # check correct up-axis
        if config_sim["up_axis"] not in ["z", "y"]:
            msg = f"Invalid physics up-axis: {config_sim['up_axis']}"
            print(msg)
            raise ValueError(msg)

        # assign general sim parameters
        sim_params.dt = config_sim["dt"]
        sim_params.num_client_threads = config_sim.get("num_client_threads", 0)
        sim_params.use_gpu_pipeline = config_sim["use_gpu_pipeline"]
        sim_params.substeps = config_sim.get("substeps", 2)

        # assign up-axis
        if config_sim["up_axis"] == "z":
            sim_params.up_axis = gymapi.UP_AXIS_Z
        else:
            sim_params.up_axis = gymapi.UP_AXIS_Y

        # assign gravity
        sim_params.gravity = gymapi.Vec3(*config_sim["gravity"])

        # configure physics parameters
        if physics_engine == "physx":
            # set the parameters
            if "physx" in config_sim:
                for opt in config_sim["physx"].keys():
                    if opt == "contact_collection":
                        setattr(
                            sim_params.physx,
                            opt,
                            gymapi.ContactCollection(config_sim["physx"][opt]),
                        )
                    else:
                        setattr(sim_params.physx, opt, config_sim["physx"][opt])
        else:
            # set the parameters
            if "flex" in config_sim:
                for opt in config_sim["flex"].keys():
                    setattr(sim_params.flex, opt, config_sim["flex"][opt])

        # return the configured params
        return sim_params

    """
    Domain Randomization methods
    """

    def get_actor_params_info(self, dr_params: Dict[str, Any], env):
        """Generate a flat array of actor params, their names and ranges.

        Returns:
            The array
        """

        if "actor_params" not in dr_params:
            return None
        params = []
        names = []
        lows = []
        highs = []
        param_getters_map = get_property_getter_map(self.gym)
        for actor, actor_properties in dr_params["actor_params"].items():
            handle = self.gym.find_actor_handle(env, actor)
            for prop_name, prop_attrs in actor_properties.items():
                if prop_name == "color":
                    continue  # this is set randomly
                props = param_getters_map[prop_name](env, handle)
                if not isinstance(props, list):
                    props = [props]
                for prop_idx, prop in enumerate(props):
                    for attr, attr_randomization_params in prop_attrs.items():
                        name = prop_name + "_" + str(prop_idx) + "_" + attr
                        lo_hi = attr_randomization_params["range"]
                        distr = attr_randomization_params["distribution"]
                        if "uniform" not in distr:
                            lo_hi = (-1.0 * float("Inf"), float("Inf"))
                        if isinstance(prop, np.ndarray):
                            for attr_idx in range(prop[attr].shape[0]):
                                params.append(prop[attr][attr_idx])
                                names.append(name + "_" + str(attr_idx))
                                lows.append(lo_hi[0])
                                highs.append(lo_hi[1])
                        else:
                            params.append(getattr(prop, attr))
                            names.append(name)
                            lows.append(lo_hi[0])
                            highs.append(lo_hi[1])
        return params, names, lows, highs

    def apply_randomizations(self, dr_params):
        """Apply domain randomizations to the environment.

        Note that currently we can only apply randomizations only on resets, due to current PhysX limitations

        Args:
            dr_params: parameters for domain randomization to use.
        """

        # If we don't have a randomization frequency, randomize every step
        rand_freq = dr_params.get("frequency", 1)

        # First, determine what to randomize:
        #   - non-environment parameters when > frequency steps have passed since the last non-environment
        #   - physical environments in the reset buffer, which have exceeded the randomization frequency threshold
        #   - on the first call, randomize everything
        self.last_step = self.gym.get_frame_count(self.sim)
        if self.first_randomization:
            do_nonenv_randomize = True
            env_ids = list(range(self.num_envs))
        else:
            do_nonenv_randomize = (self.last_step - self.last_rand_step) >= rand_freq
            rand_envs = torch.where(
                self.randomize_buf >= rand_freq,
                torch.ones_like(self.randomize_buf),
                torch.zeros_like(self.randomize_buf),
            )
            rand_envs = torch.logical_and(rand_envs, self.reset_buf)
            env_ids = torch.nonzero(rand_envs, as_tuple=False).squeeze(-1).tolist()
            self.randomize_buf[rand_envs] = 0

        if do_nonenv_randomize:
            self.last_rand_step = self.last_step

        param_setters_map = get_property_setter_map(self.gym)
        param_setter_defaults_map = get_default_setter_args(self.gym)
        param_getters_map = get_property_getter_map(self.gym)

        # On first iteration, check the number of buckets
        if self.first_randomization:
            check_buckets(self.gym, self.envs, dr_params)

        for nonphysical_param in ["observations", "actions"]:
            if nonphysical_param in dr_params and do_nonenv_randomize:
                dist = dr_params[nonphysical_param]["distribution"]
                op_type = dr_params[nonphysical_param]["operation"]
                sched_type = (
                    dr_params[nonphysical_param]["schedule"]
                    if "schedule" in dr_params[nonphysical_param]
                    else None
                )
                sched_step = (
                    dr_params[nonphysical_param]["schedule_steps"]
                    if "schedule" in dr_params[nonphysical_param]
                    else None
                )
                op = operator.add if op_type == "additive" else operator.mul

                if sched_type == "linear":
                    sched_scaling = 1.0 / sched_step * min(self.last_step, sched_step)
                elif sched_type == "constant":
                    sched_scaling = 0 if self.last_step < sched_step else 1
                else:
                    sched_scaling = 1
                if (
                    nonphysical_param == "observations"
                    and self.observation_dr_mask is not None
                ):
                    sched_scaling *= self.observation_dr_mask
                if dist == "gaussian":
                    mu, var = dr_params[nonphysical_param]["range"]
                    if hasattr(var, "__iter__"):
                        var = torch.tensor(var, device=self.device)
                    mu_corr, var_corr = dr_params[nonphysical_param].get(
                        "range_correlated", [0.0, 0.0]
                    )

                    if op_type == "additive":
                        mu *= sched_scaling
                        var *= sched_scaling
                        mu_corr *= sched_scaling
                        var_corr *= sched_scaling
                    elif op_type == "scaling":
                        var = var * sched_scaling  # scale up var over time
                        mu = mu * sched_scaling + 1.0 * (
                            1.0 - sched_scaling
                        )  # linearly interpolate

                        var_corr = var_corr * sched_scaling  # scale up var over time
                        mu_corr = mu_corr * sched_scaling + 1.0 * (
                            1.0 - sched_scaling
                        )  # linearly interpolate

                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.dr_randomizations[param_name]
                        corr = params.get("corr", None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params["corr"] = corr
                        corr = corr * params["var_corr"] + params["mu_corr"]
                        return op(
                            tensor,
                            corr
                            + torch.randn_like(tensor) * params["var"]
                            + params["mu"],
                        )

                    self.dr_randomizations[nonphysical_param] = {
                        "mu": mu,
                        "var": var,
                        "mu_corr": mu_corr,
                        "var_corr": var_corr,
                        "noise_lambda": noise_lambda,
                    }

                elif dist == "uniform":
                    lo, hi = dr_params[nonphysical_param]["range"]
                    lo_corr, hi_corr = dr_params[nonphysical_param].get(
                        "range_correlated", [0.0, 0.0]
                    )

                    if op_type == "additive":
                        lo *= sched_scaling
                        hi *= sched_scaling
                        lo_corr *= sched_scaling
                        hi_corr *= sched_scaling
                    elif op_type == "scaling":
                        lo = lo * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi = hi * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        lo_corr = lo_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi_corr = hi_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)

                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.dr_randomizations[param_name]
                        corr = params.get("corr", None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params["corr"] = corr
                        corr = (
                            corr * (params["hi_corr"] - params["lo_corr"])
                            + params["lo_corr"]
                        )
                        return op(
                            tensor,
                            corr
                            + torch.rand_like(tensor) * (params["hi"] - params["lo"])
                            + params["lo"],
                        )

                    self.dr_randomizations[nonphysical_param] = {
                        "lo": lo,
                        "hi": hi,
                        "lo_corr": lo_corr,
                        "hi_corr": hi_corr,
                        "noise_lambda": noise_lambda,
                    }

        if "sim_params" in dr_params and do_nonenv_randomize:
            prop_attrs = dr_params["sim_params"]
            prop = self.gym.get_sim_params(self.sim)

            if self.first_randomization:
                self.original_props["sim_params"] = {
                    attr: getattr(prop, attr) for attr in dir(prop)
                }

            for attr, attr_randomization_params in prop_attrs.items():
                apply_random_samples(
                    prop,
                    self.original_props["sim_params"],
                    attr,
                    attr_randomization_params,
                    self.last_step,
                )

            self.gym.set_sim_params(self.sim, prop)

        # If self.actor_params_generator is initialized: use it to
        # sample actor simulation params. This gives users the
        # freedom to generate samples from arbitrary distributions,
        # e.g. use full-covariance distributions instead of the DR's
        # default of treating each simulation parameter independently.
        extern_offsets = {}
        if self.actor_params_generator is not None:
            for env_id in env_ids:
                self.extern_actor_params[env_id] = self.actor_params_generator.sample()
                extern_offsets[env_id] = 0

        # randomise all attributes of each actor (hand, cube etc..)
        # actor_properties are (stiffness, damping etc..)

        # Loop over actors, then loop over envs, then loop over their props
        # and lastly loop over the ranges of the params
        if "actor_params" in dr_params:
            for actor, actor_properties in dr_params["actor_params"].items():
                # Loop over all envs as this part is not tensorised yet
                for env_id in env_ids:
                    env = self.envs[env_id]
                    handle = self.gym.find_actor_handle(env, actor)
                    extern_sample = self.extern_actor_params[env_id]

                    # randomise dof_props, rigid_body, rigid_shape properties
                    # all obtained from the YAML file
                    # EXAMPLE: prop name: dof_properties, rigid_body_properties, rigid_shape properties
                    #          prop_attrs:
                    #               {'damping': {'range': [0.3, 3.0], 'operation': 'scaling', 'distribution': 'loguniform'}
                    #               {'stiffness': {'range': [0.75, 1.5], 'operation': 'scaling', 'distribution': 'loguniform'}
                    for prop_name, prop_attrs in actor_properties.items():
                        if prop_name == "color":
                            num_bodies = self.gym.get_actor_rigid_body_count(
                                env, handle
                            )
                            for n in range(num_bodies):
                                self.gym.set_rigid_body_color(
                                    env,
                                    handle,
                                    n,
                                    gymapi.MESH_VISUAL,
                                    gymapi.Vec3(
                                        random.uniform(0, 1),
                                        random.uniform(0, 1),
                                        random.uniform(0, 1),
                                    ),
                                )
                            continue

                        if prop_name == "scale":
                            setup_only = prop_attrs.get("setup_only", False)
                            if (
                                setup_only and not self.sim_initialized
                            ) or not setup_only:
                                attr_randomization_params = prop_attrs
                                sample = generate_random_samples(
                                    attr_randomization_params, 1, self.last_step, None
                                )
                                og_scale = 1
                                if attr_randomization_params["operation"] == "scaling":
                                    new_scale = og_scale * sample
                                elif (
                                    attr_randomization_params["operation"] == "additive"
                                ):
                                    new_scale = og_scale + sample
                                self.gym.set_actor_scale(env, handle, new_scale)
                            continue

                        prop = param_getters_map[prop_name](env, handle)
                        set_random_properties = True

                        if isinstance(prop, list):
                            if self.first_randomization:
                                self.original_props[prop_name] = [
                                    {attr: getattr(p, attr) for attr in dir(p)}
                                    for p in prop
                                ]
                            for p, og_p in zip(prop, self.original_props[prop_name]):
                                for (
                                    attr,
                                    attr_randomization_params,
                                ) in prop_attrs.items():
                                    setup_only = attr_randomization_params.get(
                                        "setup_only", False
                                    )
                                    if (
                                        setup_only and not self.sim_initialized
                                    ) or not setup_only:
                                        smpl = None
                                        if self.actor_params_generator is not None:
                                            (
                                                smpl,
                                                extern_offsets[env_id],
                                            ) = get_attr_val_from_sample(
                                                extern_sample,
                                                extern_offsets[env_id],
                                                p,
                                                attr,
                                            )
                                        apply_random_samples(
                                            p,
                                            og_p,
                                            attr,
                                            attr_randomization_params,
                                            self.last_step,
                                            smpl,
                                        )
                                    else:
                                        set_random_properties = False
                        else:
                            if self.first_randomization:
                                self.original_props[prop_name] = deepcopy(prop)
                            for attr, attr_randomization_params in prop_attrs.items():
                                setup_only = attr_randomization_params.get(
                                    "setup_only", False
                                )
                                if (
                                    setup_only and not self.sim_initialized
                                ) or not setup_only:
                                    smpl = None
                                    if self.actor_params_generator is not None:
                                        (
                                            smpl,
                                            extern_offsets[env_id],
                                        ) = get_attr_val_from_sample(
                                            extern_sample,
                                            extern_offsets[env_id],
                                            prop,
                                            attr,
                                        )
                                    apply_random_samples(
                                        prop,
                                        self.original_props[prop_name],
                                        attr,
                                        attr_randomization_params,
                                        self.last_step,
                                        smpl,
                                    )
                                else:
                                    set_random_properties = False

                        if set_random_properties:
                            setter = param_setters_map[prop_name]
                            default_args = param_setter_defaults_map[prop_name]
                            setter(env, handle, prop, *default_args)

        if self.actor_params_generator is not None:
            for env_id in env_ids:  # check that we used all dims in sample
                if extern_offsets[env_id] > 0:
                    extern_sample = self.extern_actor_params[env_id]
                    if extern_offsets[env_id] != extern_sample.shape[0]:
                        print(
                            "env_id",
                            env_id,
                            "extern_offset",
                            extern_offsets[env_id],
                            "vs extern_sample.shape",
                            extern_sample.shape,
                        )
                        raise Exception("Invalid extern_sample size")

        self.first_randomization = False


class HardwareVecTask(VecTask):
    base_obs_list = [
        ["cubeA_pos", 3],
        ["cubeA_quat", 4],
        ["cubeA_pos_relative", 3],  # relative to EE
        ["cubeA_to_cubeB_pos", 3],
        ["cubeA_to_cubeB_rot_rad", 1],
        ["cubeB_pos", 3],
        ["cubeB_quat", 4],
        ["q_target", 8],
        ["eef_pos", 3],
        ["eef_quat", 4],
    ]
    base_obs_dict = {}
    _sum = 0
    for i in range(len(base_obs_list)):
        base_obs_dict[base_obs_list[i][0]] = [
            _sum,
            _sum + base_obs_list[i][1],
        ]
        _sum += base_obs_list[i][1]

    def __init__(
        self,
        config,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture: bool = False,
        force_render: bool = False,
    ):
        # Only 1 environment is allowed when running hardware
        self.has_hardware_action = config["run_cfg"]["hardware_action"]
        self.simulate_from_hardware_initial_state = config["run_cfg"][
            "simulate_from_hardware_initial_state"
        ]
        self.has_hardware_observation = (
            config["run_cfg"]["hardware_observation"]
            or self.simulate_from_hardware_initial_state
        )
        self.has_hardware = self.has_hardware_action or self.has_hardware_observation
        self.randomize = self.cfg["task"]["randomize"] and not self.has_hardware
        self.randomization_params = self.cfg["task"]["randomization_params"]
        # Whether to save states
        self.save_states = self.cfg["task"]["save_states"]
        self.state_history = None
        if self.save_states:
            assert config["env"]["numEnvs"] == 1
            if self.cfg["task"]["save_dir"] is None:
                self.save_dir = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "../../state_trajs",
                    str(time.time()),
                )
            else:
                self.save_dir = self.cfg["task"]["save_dir"]
            os.makedirs(self.save_dir, exist_ok=True)
            self.state_history = {}
        if self.has_hardware:
            assert config["env"]["numEnvs"] == 1
            assert config["run_cfg"]["hardware_object"] is not None
            # Disable all domain randomization
            config["task"]["randomize"] = False
        self.is_test = config["run_cfg"]["test"]

        super().__init__(
            config,
            rl_device,
            sim_device,
            graphics_device_id,
            headless,
            virtual_screen_capture,
            force_render,
        )
        # Load the controller configs
        self.joint_controller_cfg = YamlConfig(deoxys_utils.joint_controller_cfg_path)
        assert self.joint_controller_cfg["controller_type"] == "JOINT_IMPEDANCE"
        self.joint_kp_sim = torch.tensor(self.joint_controller_cfg["joint_kp_sim"]).to(
            self._q
        )
        self.joint_kd_sim = torch.tensor(self.joint_controller_cfg["joint_kd_sim"]).to(
            self._q
        )
        self.osc_controller_cfg = YamlConfig(deoxys_utils.osc_controller_cfg_path)
        assert self.osc_controller_cfg["controller_type"] == "OSC_POSE"
        Kp_translation = self.osc_controller_cfg["Kp"]["translation"]
        Kp_orientation = self.osc_controller_cfg["Kp"]["rotation"]
        self.kp = to_torch(
            [Kp_translation] * 3 + [Kp_orientation] * 3, device=self.device
        )
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = to_torch([10.0] * 7, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)

        self.max_arm_delta_q = (
            torch.tensor(self.joint_controller_cfg["max_delta_q"]).to(self._q) * 0.8
        )
        self.max_arm_delta_q_np = self.max_arm_delta_q.cpu().detach().numpy()
        # Set control limits
        self.osc_limit = np.array(self.osc_controller_cfg["max_action"])
        self.joint_imp_limit = self._franka_effort_limits[:7].cpu().detach().numpy()
        self.cmd_limit = (
            to_torch(self.osc_limit, device=self.device).unsqueeze(0)
            if self.control_type == "osc"
            else self._franka_effort_limits[:7].unsqueeze(0)
        )
        self._q_target = torch.clone(self._q)
        # Franka defaults
        self.franka_default_dof_pos = to_torch(
            [
                0.09162008114028396,
                -0.19826458111314524,
                -0.01990020486871322,
                -2.4732269941140346,
                -0.01307073642274261,
                2.30396583422025,
                0.8480939705504309,
                0.035,
                0.035,
            ],
            device=self.device,
        )
        # Code for setting primitive_sequence and target states
        # Primitive index indicates where in the primitve sequence the environment is in
        self.primitive_idx = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        # Primitive progress indicates how far along the current primitive the environment is in
        # The total progress is stored in self.progress_buf
        self.primitive_progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        # Set up Franka
        if self.has_hardware_action:
            from isaacgymenvs.utils.deoxys_utils import DeoxysRobot

            self.robot = DeoxysRobot()
        else:
            self.robot = None
        self.retargeted_primitive_sequence = None
        self.raw_primitive_sequence = None
        if self.is_test:
            if self.cfg["run_cfg"]["primitive_sequence_path"] is not None:
                # Load the primitive sequence which is a pickled object
                with open(self.cfg["run_cfg"]["primitive_sequence_path"], "rb") as f:
                    self.raw_primitive_sequence = pickle.load(f)
                # Create the drake scene
            with open(self.cfg["task"]["drake_scene_cfg_path"], "r") as cfg_file:
                self.drake_scene_cfg = yaml.safe_load(cfg_file.read())
                # update the cfg based on the wall config
                self.drake_scene_cfg = update_cfg_wall_pos_orn(
                    self.cfg["env"]["wall_left_x"],
                    self.cfg["env"]["wall_right_x"],
                    self.drake_scene_cfg,
                )
            self.object_file_path = get_tml_asset_path(
                config["env"]["chosen_objs"][0]
                if self.cfg["run_cfg"]["hardware_object"] is None
                else self.cfg["run_cfg"]["hardware_object"]
            )
        if self.has_hardware_observation:
            from contactdemo.lib.drake.state_estimator import StateEstimator

            self.state_estimator = StateEstimator(
                state_estimator_mode="megapose",
                ycb_idx=self.cfg["run_cfg"]["hardware_object"],
                visualize=False,
                filter=False,
                drake_scene_cfg=self.drake_scene_cfg,
                object_file_path=self.object_file_path,
            )
        self._update_states()

        # Set to the length from raw_primitive_sequence for now
        self.max_episode_length = (
            sum(primitive.max_num_steps for primitive in self.raw_primitive_sequence)
            if (self.is_test and self.raw_primitive_sequence is not None)
            else self.cfg["env"]["episodeLength"]
        )  #
        self.disble_ik = self.cfg["run_cfg"]["disable_ik"]
        self.visualize_retarget = self.cfg["run_cfg"]["visualize_retarget"]

    def retarget_primitives(self):
        assert self.raw_primitive_sequence is not None
        obj_X_WB = (
            self.get_hardware_object_X_WB() if self.has_hardware_observation else None
        )
        self.retargeted_primitive_sequence = retarget_primitive_sequence(
            self.raw_primitive_sequence.copy(),
            self.drake_scene_cfg,
            self.object_file_path,
            obj_X_WB,
            visualize=self.visualize_retarget,
            disable_ik=self.disble_ik,
        )
        print(
            f"{BCOLORS.OKBLUE}Primitive sequence retargeted to initial object X \n{obj_X_WB}"
        )
        for seq in self.retargeted_primitive_sequence:
            print(seq.primitive_type)
        print(BCOLORS.ENDC)
        # Update the max episode length
        self.max_episode_length = (
            sum(
                primitive.max_num_steps
                for primitive in self.retargeted_primitive_sequence
            )
            if (self.is_test and self.raw_primitive_sequence is not None)
            else self.cfg["env"]["episodeLength"]
        )  #
        # Update the target state
        for env_idx in range(self.num_envs):
            assert self.primitive_idx[env_idx] == 0
            new_primitive_goal_state = torch.tensor(
                self.retargeted_primitive_sequence[0].goal_pose_WB
            ).to(self._init_cubeB_state)
            self._init_cubeB_state[env_idx, :] = 0.0
            self._init_cubeB_state[env_idx, :7] = new_primitive_goal_state

    def apply_randomizations(self, dr_params):
        assert not self.has_hardware
        return super().apply_randomizations(dr_params)

    def _compute_sim_joint_impedance_torques(self, arm_delta_q):
        """
        On hardware this is calculated by
        tau_d_calculated[i] =
        k_gains[i] * (state.q_d[i] - state.q[i]) - d_gains[i] * state.dq[i] + coriolis[i]

        https://frankaemika.github.io/libfranka/joint_impedance_control_8cpp-example.html

        However here we neglect the coriolis term.
        """
        q, qdot = self._q[:, :7], self._qd[:, :7]  # Current state
        # clip dq
        arm_delta_q = torch.clamp(
            arm_delta_q, -self.max_arm_delta_q, self.max_arm_delta_q
        )
        # self._q_target[:, 7:] = self._q[:, 7:]
        # print("dq", dq)
        # TODO: clamp q_target?
        u = self.joint_kp_sim * arm_delta_q - self.joint_kd_sim * qdot
        # clamp the actions
        # print("u", u)
        u = tensor_clamp(
            u.squeeze(-1),
            -self._franka_effort_limits[:7].unsqueeze(0),
            self._franka_effort_limits[:7].unsqueeze(0),
        )
        return u

    def _compute_osc_torques(self, dpose):
        # Solve for Operational Space Control # Paper: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        # Helpful resource: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
        q, qd = self._q[:, :7], self._qd[:, :7]
        mm_inv = torch.inverse(self._mm)
        m_eef_inv = self._j_eef @ mm_inv @ torch.transpose(self._j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)

        # Transform our cartesian action `dpose` into joint torques `u`
        u = (
            torch.transpose(self._j_eef, 1, 2)
            @ m_eef
            @ (self.kp * dpose - self.kd * self.states["eef_vel"]).unsqueeze(-1)
        )

        # Nullspace control torques `u_null` prevents large changes in joint configuration
        # They are added into the nullspace of OSC so that the end effector orientation remains constant
        # roboticsproceedings.org/rss07/p31.pdf
        j_eef_inv = m_eef @ self._j_eef @ mm_inv
        u_null = self.kd_null * -qd + self.kp_null * (
            (self.franka_default_dof_pos[:7] - q + np.pi) % (2 * np.pi) - np.pi
        )
        u_null[:, 7:] *= 0
        u_null = self._mm @ u_null.unsqueeze(-1)
        u += (
            torch.eye(7, device=self.device).unsqueeze(0)
            - torch.transpose(self._j_eef, 1, 2) @ j_eef_inv
        ) @ u_null
        # Clip the values to be within valid effort range
        u = tensor_clamp(
            u.squeeze(-1),
            -self._franka_effort_limits[:7].unsqueeze(0),
            self._franka_effort_limits[:7].unsqueeze(0),
        )
        return u

    def pre_physics_step(self, actions):
        """
        Apply actions. This is shared between all extrinsic manipulation environments
        Returns arm_actions(delta_q or delta_pose) and gripper_actions
        :return arm_action, gripper_action
        """
        raise AssertionError("pre_physics_step should not be called on HardwareVecTask")
        # Actions are between -1~1
        self.actions = actions.clone().to(self.device)

        # Split arm and gripper command
        arm_actions, gripper_actions = self.actions[:, :-1], self.actions[:, -1]
        # map [-1, 1] to [0., 0.08]
        gripper_actions = (gripper_actions + 1.0) * self.franka_dof_upper_limits[
            -2
        ].item()  # lower limit is 0., upper limit is 0.08 (*2 fingers)
        # Thus gripper_action is between 0~0.08
        # Control arm (scale value first)
        if self.control_type == "osc":
            dpose = self.cmd_limit * arm_actions
            self.control_sim_robot(dpose, gripper_actions, "osc")
        elif self.control_type == "joint_imp":
            # Scale to be within dq
            arm_actions = arm_actions * self.max_arm_delta_q
            self.control_sim_robot(arm_actions, gripper_actions, "joint_imp")
        else:
            raise NotImplementedError
        return arm_actions, gripper_actions

    def get_hardware_object_X_WB(
        self,
        R_WB_guess=None,
        p_WB_guess=None,
        filter=False,
        run_global_registration=False,
    ):
        """
        X_RB is the pose of the object in the robot frame. All camera poses are in the robot frame.
        X_WB is the pose of the object in the world frame
        X_WB = X_WR * X_RB
        """
        X_RW = np.linalg.inv(X_WR)
        R_RB_guess = None
        p_RB_guess = None
        if R_WB_guess is not None:
            R_RB_guess = X_RW[:3, :3] @ R_WB_guess
        if p_WB_guess is not None:
            p_RB_guess = X_RW[:3, :3] @ (p_WB_guess) + X_RW[:3, 3]
        X_RB = self.state_estimator.get_object_X_RB(
            R_RB_guess=R_RB_guess,
            p_RB_guess=p_RB_guess,
            run_global_registration=run_global_registration,
            filter=filter,
        )
        X_WB = X_WR @ X_RB
        return X_WB

    def reset_hardware_robot(self):
        assert self.has_hardware_action
        print("Resetting robot")
        self.robot.set_q_to_default()
        print("Robot resetting done")
        self._q_target[:, :7] = torch.tensor(self.robot.last_arm_q, device=self.device)
        # Override robot arm states with hardware values
        multi_env_ids_int32 = self._global_indices[0, 0].flatten()
        self._q[:, :7] = torch.from_numpy(self.robot.last_arm_q).to(self._q)
        self._q[:, 7:] = self.robot.last_gripper_q * 0.08 / 2.0
        # Set sim tensors to the hardware values
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        If self.has_hardware, actuate the robot with the given actions.

        Args:
            actions: actions to apply as specified by the pushing primitive
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """
        # randomize actions
        if self.dr_randomizations.get("actions", None) and not self.has_hardware:
            # Don't domain randomize hardware actions
            actions = self.dr_randomizations["actions"]["noise_lambda"](actions)
        # Action is 8D (7 arm joints + 1 gripper) and is within -1~1

        ####### For policy actions #######
        # The clipping shouldn't really do anything since self.clip_actions is 1
        policy_action_tensor = torch.clamp(
            actions, -self.clip_actions, self.clip_actions
        )
        # Actions are between -1~1
        # Split arm and gripper command
        policy_arm_actions, policy_gripper_actions = (
            policy_action_tensor[:, :-1],
            policy_action_tensor[:, -1],
        )
        # map [-1, 1] to [0., 0.08]
        # Don't allow policy to move gripper due to gripper control issues
        policy_gripper_actions[:] = 0.08
        # policy_gripper_actions = (
        #     policy_gripper_actions + 1.0
        # ) * self.franka_dof_upper_limits[
        #     -2
        # ].item()  # lower limit is 0., upper limit is 0.08 (*2 fingers)
        # Thus gripper_action is between 0~0.08
        # Control arm (scale value first)
        current_control_type = self.control_type
        u_arm = torch.zeros_like(self._arm_control)
        u_fingers = torch.zeros_like(self._gripper_control)
        if self.is_test and self.retargeted_primitive_sequence is not None:
            primitive_arm_actions = []
            primitive_gripper_actions = []
            # The override is only done in "test" mode
            for env_idx in range(self.num_envs):
                if (
                    self.retargeted_primitive_sequence[
                        self.primitive_idx[env_idx]
                    ].primitive_type
                    == primitive_types.PrimitiveType.Push
                ):
                    # Check the control type
                    if self.control_type == "joint_imp":
                        policy_arm_actions = policy_arm_actions * self.max_arm_delta_q
                    elif self.control_type == "osc":
                        policy_arm_actions = self.cmd_limit * policy_arm_actions
                    else:
                        raise NotImplementedError
                    arm_action, gripper_action = (
                        policy_arm_actions[env_idx].cpu().detach().numpy(),
                        policy_gripper_actions[env_idx].cpu().detach().numpy(),
                    )
                else:
                    # Get new actions from the primitive controller
                    arm_action, gripper_action = self.retargeted_primitive_sequence[
                        self.primitive_idx[env_idx]
                    ].controller.compute_action(
                        state=self.states,
                        current_step=self.primitive_progress_buf[env_idx]
                        .detach()
                        .cpu()
                        .numpy(),
                        env_idx=env_idx,
                        current_primitive=self.retargeted_primitive_sequence[
                            self.primitive_idx[env_idx]
                        ],
                    )

                # Only push is learned. Recompute the actions if it's not push
                current_control_type = self.retargeted_primitive_sequence[
                    self.primitive_idx[env_idx]
                ].control_type
                # gripper action should be a scalar between 0 and 0.08
                # assert 0 <= gripper_action <= 0.08
                if current_control_type == "osc":
                    if np.any(arm_action < -self.osc_limit - 0.0001) or np.any(
                        arm_action > self.osc_limit + 0.0001
                    ):
                        print(
                            f"vec_task OSC limit violated, action out of bounds: {arm_action}"
                        )
                    # arm_action = np.clip(arm_action, -self.osc_limit, self.osc_limit)
                    u_arm[env_idx, :] = self._compute_osc_torques(
                        dpose=torch.tensor(
                            arm_action, device=self.device, dtype=torch.float
                        )
                    )
                elif current_control_type == "joint_imp":
                    # Note that joint_imp is delta_q
                    # Scale to be within dq
                    if np.any(arm_action < -self.max_arm_delta_q_np - 0.0001) or np.any(
                        arm_action > self.max_arm_delta_q_np + 0.0001
                    ):
                        print(
                            f"vec_task joint_imp limit violated, action out of bounds: {arm_action}"
                        )
                        breakpoint()
                    # arm_action = np.clip(
                    #     arm_action, -self.joint_imp_limit, self.joint_imp_limit
                    # )
                    arm_action = torch.tensor(
                        arm_action, device=self.device, dtype=torch.float
                    )
                    u_arm[env_idx, :] = self._compute_sim_joint_impedance_torques(
                        arm_delta_q=arm_action
                    )
                    # Update self._q_target
                    self._q_target[env_idx, :7] = (
                        self.states["q"][env_idx, :7] + arm_action
                    )
                elif current_control_type == "joint_target":
                    # joint_target is a set of angles the arm should go to
                    # Don't clip and just set directly to the target in sim
                    # The real robot handles it with deoxys
                    # TODO
                    arm_action = torch.tensor(
                        arm_action, device=self.device, dtype=torch.float
                    )
                    u_arm[env_idx, :] = self._compute_sim_joint_impedance_torques(
                        arm_delta_q=arm_action - self.states["q"][env_idx, :7]
                    )
                    # Update self._q_target
                    self._q_target[env_idx, :7] = arm_action
                else:
                    raise NotImplementedError
                primitive_arm_actions.append(arm_action)
                primitive_gripper_actions.append(gripper_action)
                u_fingers[env_idx, :] = gripper_action / 2.0
                self._q_target[env_idx, 7:9] = gripper_action / 2.0
        else:
            if self.control_type == "osc":
                policy_arm_actions = self.cmd_limit * policy_arm_actions
                u_arm = self._compute_osc_torques(dpose=policy_arm_actions)
            elif self.control_type == "joint_imp":
                # Scale to be within dq
                policy_arm_actions = policy_arm_actions * self.max_arm_delta_q
                u_arm = self._compute_sim_joint_impedance_torques(
                    arm_delta_q=policy_arm_actions
                )
                # Update self._q_target
                self._q_target[:, :7] = self.states["q"][:, :7] + policy_arm_actions
                self._q_target[:, 7] = policy_gripper_actions / 2.0
                self._q_target[:, 8] = policy_gripper_actions / 2.0
            else:
                # Note that joint_target only appears in teh context of primitives
                raise NotImplementedError
            # Note self.franka_dof_upper_limits[-2].item() = 0.04
            u_fingers[:, :] = policy_gripper_actions.view(-1, 1) / 2.0

        ##################################
        # breakpoint()  # check what states is and whether it matches hw
        if self.save_states:
            raise DeprecationWarning
            # Save action
            # Note that saving only works with 1 env for now
            timestep = self.progress_buf.detach().cpu().numpy()[0]
            if timestep not in self.state_history:
                self.state_history[timestep] = {}
            self.state_history[timestep]["arm_actions"] = (
                policy_arm_actions.detach().cpu().numpy()
            )
            self.state_history[timestep]["gripper_actions"] = (
                policy_gripper_actions.detach().cpu().numpy()
            )

        # step physics and render each frame
        for i in range(self.control_freq_inv):
            self.control_sim_robot(u_arm, u_fingers)
            if self.force_render:
                self.render()
            self.gym.simulate(self.sim)
            self._refresh_sim_tensors()  # Refresh on every step
            if self.has_hardware_action:
                break  # need to run at least 1 step of simulation to set the state tensors
            if self.is_test and self.retargeted_primitive_sequence is not None:
                # Control type may not be all equal
                # FIXME: the logic of this is wrong
                for env_idx in range(self.num_envs):
                    current_control_type = self.retargeted_primitive_sequence[
                        self.primitive_idx[env_idx]
                    ].control_type
                    if (
                        current_control_type == "joint_imp"
                        or current_control_type == "joint_target"
                    ):
                        # in simulation mode, the impedance control torque should be refreshed multiple times
                        arm_delta_q = (
                            self._q_target[env_idx, :7] - self.states["q"][env_idx, :7]
                        )
                        u_arm = self._compute_sim_joint_impedance_torques(
                            arm_delta_q=arm_delta_q
                        )
                    elif current_control_type == "osc":
                        # FIXME: this shouldn't be necessary
                        u_arm[env_idx, :] = self._compute_osc_torques(
                            dpose=torch.tensor(
                                primitive_arm_actions[env_idx],
                                device=self.device,
                                dtype=torch.float,
                            )
                        )
            else:
                # Load in vectorized fashion since there is only 1 primitive type (learned policy)
                if self.joint_controller_cfg["controller_type"] == "JOINT_IMPEDANCE":
                    u_arm = self._compute_sim_joint_impedance_torques(
                        arm_delta_q=self._q_target[:, :7] - self.states["q"][:, :7]
                    )
                else:
                    # Compute OSC torque
                    u_arm[env_idx, :] = self._compute_osc_torques(
                        dpose=torch.tensor(
                            primitive_arm_actions[env_idx],
                            device=self.device,
                            dtype=torch.float,
                        )
                    )
        # By now self.states["q"] should be the sim values
        # send hardware command
        # Run the hardware command
        if self.has_hardware_action:
            # Note that the states have not been updated at this point,
            # so it is identical to before the simulation step (and applying the action)
            # The states are refreshed after self._refresh_sim_tensors()
            if self.retargeted_primitive_sequence is not None:
                assert len(primitive_arm_actions) == 1
                assert len(primitive_gripper_actions) == 1
                hw_gripper_open_ratio = primitive_gripper_actions[0] / 0.08
                if isinstance(primitive_arm_actions[0], torch.Tensor):
                    hw_arm_action = (
                        primitive_arm_actions[0].squeeze().detach().cpu().numpy()
                    )
                else:
                    hw_arm_action = primitive_arm_actions[0]

            else:
                # policy_gripper_actions is between 0~0.08
                hw_gripper_open_ratio = (
                    policy_gripper_actions.squeeze().detach().cpu().numpy()
                ) / 0.08
                hw_arm_action = policy_arm_actions.squeeze().detach().cpu().numpy()
            # assert 0 <= hw_gripper_open_ratio <= 1

            if self.cfg["env"]["controlType"] == "joint_tor":
                raise DeprecationWarning
                q_sim = self.states["q"].detach().cpu().numpy()[0]
                arm_q_target = q_sim[:7] + policy_action_tensor[:7]
                hardware_target_pos[:7] = arm_q_target
                self.robot.set_q_to(hardware_target_pos)
                time.sleep(0.1)
            elif current_control_type == "joint_imp":
                hardware_target_pos = np.zeros(8)
                # Set the target state directly
                # Note that q_target is set by q + scaled action in pre_physics_step,
                # so there is no need to scale the action again
                hardware_target_pos[:7] = (
                    # self.states["q_target"][:, :7].detach().cpu().numpy()
                    hw_arm_action
                    + self.robot.last_arm_q
                )
                hardware_target_pos[7] = (
                    hw_gripper_open_ratio  # Note that this takes 0~0.08
                )
                self.hardware_robot_q = torch.tensor(
                    self.robot.joint_imp_action(hardware_target_pos, 10, 2, "min_jerk")
                ).to(self.device)
            elif current_control_type == "osc":
                # OSC displacements are stored in arm_action, gripper_action
                # OSC command is formatted as [dx, dy, dz, droll, dpitch, dyaw, gripper]
                # where gripper is between 0~1 (0~0.08cm distance)
                hardware_action = np.zeros(7)
                hardware_action[:6] = hw_arm_action
                # FIXME
                hardware_action[6] = hw_gripper_open_ratio
                self.robot.robot_interface.control(
                    controller_type="OSC_POSE",
                    action=hardware_action,
                    controller_cfg=self.robot.osc_controller_cfg,
                )
            elif current_control_type == "joint_target":
                # Give a joint target and let Deoxys handle the rest
                q_target = np.zeros(8)
                q_target[:7] = hw_arm_action
                q_target[7] = hw_gripper_open_ratio
                self.robot.set_q_to(q_target[:8], arm_tol=0.05)
            # Override robot arm states with hardware values
            multi_env_ids_int32 = self._global_indices[0, 0].flatten()
            self._q[:, :7] = torch.from_numpy(self.robot.last_arm_q).to(self._q)
            self._q[:, 7:] = self.robot.last_gripper_q * 0.08 / 2.0
            # Set sim tensors to the hardware values
            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self._dof_state),
                gymtorch.unwrap_tensor(multi_env_ids_int32),
                len(multi_env_ids_int32),
            )
            # Save the robot states
            if self.save_states:
                timestep = self.progress_buf.detach().cpu().numpy()[0]
                if timestep not in self.state_history:
                    self.state_history[timestep] = {}
                self.state_history[timestep][
                    "hw_eef_quat_and_pos"
                ] = self.robot.last_eef_quat_and_pos
                self.state_history[timestep]["hw_eef_pose"] = self.robot.last_eef_pose
                self.state_history[timestep]["hw_last_arm_q"] = self.robot.last_arm_q

                self.state_history[timestep][
                    "hw_last_gripper_q"
                ] = self.robot.last_gripper_q
        if self.has_hardware_observation:
            # Obtain hardware observations after refreshing the sim tensor
            # to avoid overriding the hardware values
            # compute the initial guess from the previous object pose
            # R_WB_guess = quat_to_R(self.states["cubeB_quat"].detach().cpu().numpy())
            X_WB = self.get_hardware_object_X_WB(
                filter=False, run_global_registration=False
            )
            hardware_cubeA_pos = X_WB[:3, 3]
            hardware_cubeA_quat = R.from_matrix(X_WB[:3, :3]).as_quat()
            # Override the object state in simulation
            # Note that modifying self._cubeA_state changes self._root_state
            self._cubeA_state[:, :3] = torch.from_numpy(hardware_cubeA_pos).to(
                self._cubeA_state
            )
            self._cubeA_state[:, 3:7] = torch.from_numpy(hardware_cubeA_quat).to(
                self._cubeA_state
            )
            multi_env_ids_cubes_int32 = self._global_indices[0, -2:].flatten()
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self._root_state),
                gymtorch.unwrap_tensor(multi_env_ids_cubes_int32),
                len(multi_env_ids_cubes_int32),
            )
        # compute observations, rewards, resets, ...
        self.progress_buf += 1
        self.primitive_progress_buf += 1
        self.post_physics_step()
        # Recompute reset condition for environments that are not running the learned policy
        if self.is_test and self.retargeted_primitive_sequence is not None:
            # The override is only done in "test" mode
            for env_idx in range(self.num_envs):
                # FIXME: this is a hack to disable pushing success criterion when there are multiple primitives
                self.reset_buf[env_idx] = 0
                if (
                    self.retargeted_primitive_sequence[
                        self.primitive_idx[env_idx]
                    ].primitive_type
                    == primitive_types.PrimitiveType.Push
                ):
                    is_success = self.success_buf[env_idx]
                else:
                    # First override the reset_buf as the pushing success criterion is not used
                    is_success = self.retargeted_primitive_sequence[
                        self.primitive_idx[env_idx]
                    ].controller.is_success(
                        state=self.states,
                        current_step=self.primitive_progress_buf[env_idx]
                        .cpu()
                        .detach(),
                        env_idx=env_idx,
                    )
                if is_success:
                    # Advance the primitive count and reset the progress
                    print(
                        f"{BCOLORS.OKCYAN}Env {env_idx} succeeded primitive {self.primitive_idx[env_idx]}: {self.retargeted_primitive_sequence[self.primitive_idx[env_idx]].primitive_type}{BCOLORS.ENDC}"
                    )
                    self.primitive_progress_buf[env_idx] = 0
                    self.primitive_idx[env_idx] += 1
                    self.success_buf[env_idx] = 0  # Override success buf
                    # Reset the primitive object
                    # Check if we reached the end of the primitive sequence
                    if self.primitive_idx[env_idx] >= len(
                        self.retargeted_primitive_sequence
                    ):
                        # Reset the primitive index
                        self.primitive_idx[env_idx] = 0
                        # Reset the environment
                        self.reset_buf[env_idx] = 1
                        # Reset the overall progress
                        self.progress_buf[env_idx] = 0
                        print(
                            f"{BCOLORS.OKGREEN}Env {env_idx} succeeded all primitives{BCOLORS.ENDC}"
                        )
                    # TODO: should this be included?
                    else:
                        self.retargeted_primitive_sequence[
                            self.primitive_idx[env_idx]
                        ].start_pose_WB = (
                            X_to_state(self.get_hardware_object_X_WB())
                            if self.has_hardware_observation
                            else self._cubeA_state[env_idx, :7].detach().cpu().numpy()
                        )
                        if self.has_hardware_action:
                            # Set the current robot q
                            q = np.zeros(9)
                            q[:7] = self.robot.last_arm_q
                            q[7:] = self.robot.last_gripper_q / 2.0
                            self.retargeted_primitive_sequence[
                                self.primitive_idx[env_idx]
                            ].start_robot_q = q
                        else:
                            self.retargeted_primitive_sequence[
                                self.primitive_idx[env_idx]
                            ].start_robot_q = (self._q[0].detach().cpu().numpy())
                        time.sleep(1)
                        self.retargeted_primitive_sequence[
                            self.primitive_idx[env_idx]
                        ] = update_primitive_given_start_WB_change(
                            self.retargeted_primitive_sequence[
                                self.primitive_idx[env_idx]
                            ],
                            scene_cfg=self.drake_scene_cfg,
                            object_file=self.object_file_path,
                        )

                        if (
                            self.retargeted_primitive_sequence[
                                self.primitive_idx[env_idx]
                            ].goal_pose_WB
                            is not None
                        ):
                            # Set cubeB to the next primitive's target state for visualization
                            # Note that modifying self._cubeA_state changes self._root_state
                            new_primitive_goal_state = torch.tensor(
                                self.retargeted_primitive_sequence[
                                    self.primitive_idx[env_idx]
                                ].goal_pose_WB
                            ).to(self._cubeB_state)
                            self._cubeB_state[env_idx, :] = 0.0
                            self._cubeB_state[env_idx, :7] = new_primitive_goal_state
                    print(
                        f"{BCOLORS.OKBLUE}Env {env_idx} starting primitive {self.primitive_idx[env_idx]}: {self.retargeted_primitive_sequence[self.primitive_idx[env_idx]].primitive_type}{BCOLORS.ENDC}"
                    )

                # Check for primitive timeout
                elif (
                    self.primitive_progress_buf[env_idx]
                    >= self.retargeted_primitive_sequence[
                        self.primitive_idx[env_idx]
                    ].max_num_steps
                ) or (self.progress_buf[env_idx] >= self.max_episode_length - 1):
                    print(
                        "{}Env {} failed at primitive {}: {}{}".format(
                            BCOLORS.FAIL,
                            env_idx,
                            self.primitive_idx[env_idx],
                            self.retargeted_primitive_sequence[
                                self.primitive_idx[env_idx]
                            ].primitive_type,
                            BCOLORS.ENDC,
                        )
                    )
                    # Reset the primitive count and reset the progress
                    self.primitive_progress_buf[env_idx] = 0
                    self.primitive_idx[env_idx] = 0
                    # Reset the overall progress
                    self.progress_buf[env_idx] = 0
                    # Reset the environment
                    self.reset_buf[env_idx] = 1
                # Check for pushing
                elif (
                    self.retargeted_primitive_sequence[
                        self.primitive_idx[env_idx]
                    ].primitive_type
                    == primitive_types.PrimitiveType.Push
                    and self.primitive_progress_buf[env_idx]
                    > self.cfg["env"]["episodeLength"]
                ):
                    print(
                        "{}Env {} failed at primitive {}: {}{}".format(
                            BCOLORS.FAIL,
                            env_idx,
                            self.primitive_idx[env_idx],
                            self.retargeted_primitive_sequence[
                                self.primitive_idx[env_idx]
                            ].primitive_type,
                            BCOLORS.ENDC,
                        )
                    )  # Reset the primitive count and reset the progress
                    self.primitive_progress_buf[env_idx] = 0
                    self.primitive_idx[env_idx] = 0
                    # Reset the overall progress
                    self.progress_buf[env_idx] = 0
                    # Reset the environment
                    self.reset_buf[env_idx] = 1
            # Update the root state to reflect any _cubeB_state changes
            multi_env_ids_cubes_int32 = self._global_indices[0, -2:].flatten()
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self._root_state),
                gymtorch.unwrap_tensor(multi_env_ids_cubes_int32),
                len(multi_env_ids_cubes_int32),
            )
        # Check for any resets now that reset_buf has been updated
        # FIXME: this was originally in post_physics_step()
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        #     # refresh observation if anything has been reset
        self.compute_observations()
        self.compute_reward()
        # randomize observations
        if self.dr_randomizations.get("observations", None):
            self.obs_buf = self.dr_randomizations["observations"]["noise_lambda"](
                self.obs_buf
            )
        # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 1. Timeout == 1 makes sense only if the reset buffer is 1.
        self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (
            self.reset_buf != 0
        )

        # randomize observations
        # FIXME: make this cleaner
        """
        obs = [
            "cubeA_pos",
            "cubeA_quat",
            "cubeA_pos_relative",
            "cubeA_to_cubeB_pos",
            "cubeA_to_cubeB_rot_rad",
            "cubeB_pos",
            "cubeB_quat",
            "q_target",
            "eef_pos",
            "eef_quat",
        ]
        """
        if self.randomization_params["hw_vec_task_randomize"] and not self.has_hardware:
            raise DeprecationWarning
            # Randomize cubeA_pos
            pos_sigma = self.randomization_params["observations"]["range_pos"][1]
            self.obs_buf[:, :3] = (
                self.obs_buf[:, :3] + torch.randn_like(self.obs_buf[:, :3]) * pos_sigma
            )
            # Randomize cubeA_quat
            rot_sigma = self.randomization_params["observations"]["range_rot"][1]
            # apply rotation
            r, p, y = get_euler_xyz(self.obs_buf[:, 3:7])
            r = r + torch.randn_like(r) * rot_sigma
            p = p + torch.randn_like(p) * rot_sigma
            y = y + torch.randn_like(y) * rot_sigma
            # recompute cubeA_quat
            self.obs_buf[:, 3:7] = quat_from_euler_xyz(r, p, y)
            # recompute cubeA_to_cubeB_pos
            self.obs_buf[:, 7:10] = self._cubeB_state[:, :3] - self.obs_buf[:, :3]
            # recompute cubeA_to_cubeB_rot_rad
            self.obs_buf[:, 10:13] = quat_diff_rad(
                self.obs_buf[:, 3:7], self._cubeB_state[:, 3:7]
            ).reshape(-1, 1)
        # if self.dr_randomizations.get("observations", None):
        #     self.obs_buf = self.dr_randomizations["observations"]["noise_lambda"](
        #         self.obs_buf
        #     )

        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)

        self.obs_dict["obs"] = torch.clamp(
            self.obs_buf, -self.clip_obs, self.clip_obs
        ).to(self.rl_device)
        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return (
            self.obs_dict,
            self.rew_buf.to(self.rl_device),
            self.reset_buf.to(self.rl_device),
            self.extras,
        )

    def save_and_clear_state_history(self):
        with open(
            os.path.join(
                self.save_dir,
                f"state_hist_hw_init_{self.simulate_from_hardware_initial_state}_obs_{self.has_hardware_observation}_act_{self.has_hardware_action}_{time.time()}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(self.state_history, f)
        self.state_history = {}

    """
    The following functions are specific to multi-stage task inference,
    to be called by 
    """

    def set_cube_state(
        self,
        cubeA_pos=None,
        cubeA_quat=None,
        cubeB_pos=None,
        cubeB_quat=None,
    ):
        """ """
        # Note that modifying self._cubeA_state changes self._root_state
        if cubeA_pos is not None:
            self._cubeA_state[:, :3] = torch.from_numpy(cubeA_pos).to(self._cubeA_state)
        if cubeA_quat is not None:
            self._cubeA_state[:, 3:7] = torch.from_numpy(cubeA_quat).to(
                self._cubeA_state
            )
        if cubeB_pos is not None:
            self._cubeB_state[:, :3] = torch.from_numpy(cubeB_pos).to(self._cubeB_state)
        if cubeB_quat is not None:
            self._cubeB_state[:, 3:7] = torch.from_numpy(cubeB_quat).to(
                self._cubeB_state
            )

    def control_sim_robot(self, u_arm, u_fingers):
        """
        if self.control_type == "osc":
            dpose = self.cmd_limit * arm_actions
            u_arm = self._compute_osc_torques(dpose=dpose)
        elif self.control_type == "joint_imp":
            # Scale to be within dq
            arm_actions = arm_actions * self.max_arm_delta_q
            u_arm = self._compute_sim_joint_impedance_torques(arm_delta_q=arm_actions)
        else:
            raise NotImplementedError
        self._arm_control[:, :] = u_arm
        self._gripper_control[:, :] = u_fingers
        """
        self._arm_control[:, :] = u_arm
        self._gripper_control[:, :] = u_fingers
        # Deploy actions
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self._pos_control)
        )
        self.gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(self._effort_control)
        )
