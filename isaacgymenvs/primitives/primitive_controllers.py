import isaacgymenvs.utils.math_utils as math_utils
import numpy as np
from scipy.spatial.transform import Rotation

import abc
from abc import ABC
import torch
import time
import os


current_path = os.path.dirname(os.path.realpath(__file__))
osc_controller_cfg_path = os.path.join(
    current_path, "../cfg/franka/osc-pose-controller.yml"
)
try:
    from deoxys.utils.config_utils import get_controller_config_from_file
    osc_controller_cfg = get_controller_config_from_file(osc_controller_cfg_path)
    osc_max_action = np.array(osc_controller_cfg["max_action"])
except ImportError:
    print("Import deoxys failed. Unable to run on hardware.")

class BaseController(ABC):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def compute_action(self, *args, **kwargs):
        """Compute the next step action
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
        if self.control_type == "osc":
            obs += ["q_gripper"]
        elif self.control_type == "joint_tor" or self.control_type == "joint_imp":
            obs += ["q"]
        if self.cfg["task"]["use_latent"]:
            obs += ["cubeA_latent"]

        gripper action should be a scalar between 0 and 0.08
        return arm_action, gripper_action, control_type
        """

    @abc.abstractmethod
    def is_success(self, *args, **kwargs):
        """Check if the controller has reached the goal"""

    @abc.abstractmethod
    def clear(self, *args, **kwargs):
        """Clear the controller of any memory"""


"""
States stored
"q": self._q[:, :],
"q_gripper": self._q[:, -2:],
"eef_pos": self._eef_state[:, :3],
"eef_quat": self._eef_state[:, 3:7],
"eef_vel": self._eef_state[:, 7:],
"eef_lf_pos": self._eef_lf_state[:, :3],
"eef_rf_pos": self._eef_rf_state[:, :3],
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
]  # note that last 2 entries are identical
"""


class PivotController(BaseController):
    def __init__(self, action_steps, scene_cfg, shrink_factor=0.95) -> None:
        """
        X_WE_start: the initial pose of the end effector
        p_WC: the center of the rotation
        action_steps: the number of steps to complete the pivot
        shrink_factor: the factor to shrink the "b" parameter in the Trammel of Archimedes
        """
        # TODO: figure out what is a good way to implement this
        super().__init__()
        self.action_steps = action_steps
        self.default_shrink_factor = shrink_factor
        self.max_axial_correction = 0.02
        self.max_tangent_correction = 0.5
        self.axial_err_target = 0.05
        self.axial_error_gain = 0.4
        self.tangent_gain = 0.2
        self.steps_to_withdraw = 25
        self.wall_face_center_x = (
            np.average([scene_cfg["wall"]["left_x"], scene_cfg["wall"]["right_x"]])
            + scene_cfg["franka"]["franka_pos"][0]
        )
        self.wall_yaw = scene_cfg["wall"]["wall_rpy"][-1]
        # The rotation matrix associated with the wall yaw
        self.R_wall = Rotation.from_euler("z", self.wall_yaw).as_matrix()
        self.wall_normal = self.R_wall @ np.array([1, 0, 0])
        self.push_axis = np.array(
            [1, 0, 0.25]
        )  # np.array([1.0, 0.0, 0.0])  # np.array([1.0, 0.0, -0.25])
        self.push_axis /= np.linalg.norm(self.push_axis)
        self.push_axis = self.R_wall @ self.push_axis
        self.table_z_coord = (
            scene_cfg["table"]["table_pos"][2]
            + scene_cfg["table"]["table_dims"][2] / 2.0
        )
        self.clear()

    def compute_action(self, state, current_step, env_idx, *args, **kwargs):
        """
        state: torch tensor with
        return:
        """
        # Need eef_pose, cubeA_pos
        eef_pose = np.zeros(7)
        eef_pose[:3] = (
            state["eef_pos"][env_idx].detach().cpu().numpy()
            if isinstance(state["eef_pos"], torch.Tensor)
            else state["eef_pos"][env_idx]
        )
        eef_pose[3:] = (
            state["eef_quat"][env_idx].detach().cpu().numpy()
            if isinstance(state["eef_quat"], torch.Tensor)
            else state["eef_quat"][env_idx]
        )
        X_WE = math_utils.state_to_X(eef_pose)
        # TODO: move this

        if current_step == 0:
            # First squeeze toward the object
            # Decide squeeze force based on the offset between
            self.done_squeezing = False
            self.initialized_pivot = False
            self.pivot_start_step = None
            self.osc_pos_init = eef_pose[:3].copy()
            self.last_osc_pos_target = self.osc_pos_init.copy()
            self.done = False
            self.withdraw_steps = 0
            self.withdrawing = False
            # Compute OSC tracking error to decide if squeezing is done
        osc_pos_err = eef_pose[:3] - self.last_osc_pos_target
        gripper_action = (
            np.sum(state["q"][env_idx, 7:9].detach().cpu().numpy())
            if isinstance(state["q"], torch.Tensor)
            else np.sum(state["q"][env_idx, 7])
        )

        if not self.done_squeezing:
            self.done_squeezing = np.linalg.norm(osc_pos_err) > 0.03  # to tune

            # Push the end effector toward the object
            dpose = np.zeros(6)
            self.last_osc_pos_target += 0.005 * self.push_axis  # Push forward
            dpose[:3] = self.last_osc_pos_target - eef_pose[:3]
            # breakpoint()
            return dpose, gripper_action

        if self.done_squeezing and not self.initialized_pivot:
            self.pivot_start_step = current_step
            print("Done squeezing, starting pivot")
            # Initialize pivot
            # Position of the object center in world
            self.X_WE_start = X_WE.copy()
            # b is the vector from the object's center to the wall
            self.b = np.dot(
                np.array([self.wall_face_center_x, 0.0]) - X_WE[0:2, 3],
                self.wall_normal[:2],
            )
            self.a = (
                self.X_WE_start[2, 3] - self.table_z_coord
            ) * 2.0  # assume gripper is at half of the object height
            print("a, b", self.a, self.b)
            self.rotation_center = self.X_WE_start[:3, 3].copy()
            self.rotation_center += self.b * self.wall_normal
            self.rotation_center[2] = self.table_z_coord
            self.initialized_pivot = True
            self.initial_obj_rot_inv = (
                Rotation.inv(
                    Rotation.from_quat(
                        state["cubeA_quat"][env_idx].cpu().detach().numpy()
                    )
                )
                if isinstance(state["cubeA_quat"], torch.Tensor)
                else Rotation.inv(Rotation.from_quat(state["cubeA_quat"][env_idx]))
            )
            self.initial_ee_rot_inv = Rotation.inv(
                Rotation.from_matrix(self.X_WE_start[:3, :3])
            )
        current_rot = Rotation.from_quat(
            state["cubeA_quat"][env_idx].detach().cpu().numpy()
            if isinstance(state["cubeA_quat"], torch.Tensor)
            else state["cubeA_quat"][env_idx]
        )
        rot_diff = self.initial_obj_rot_inv * current_rot
        print("rot_diff", np.linalg.norm(Rotation.as_rotvec(rot_diff)))
        if (
            np.linalg.norm(Rotation.as_rotvec(rot_diff)) > np.pi / 2.0 - 0.15
            and current_step - self.pivot_start_step > self.action_steps * 0.7
            or self.withdrawing
        ):
            print("withdrawing")
            self.withdrawing = True
            if self.withdraw_steps == self.steps_to_withdraw:
                self.done = True
                return np.zeros(6), gripper_action

            self.withdraw_steps += 1
            dpose = np.zeros(6)
            dpose[2] = 0.01
            return dpose, gripper_action

        t = (current_step - self.pivot_start_step) / self.action_steps
        theta = np.minimum(np.pi / 2 * t, np.pi / 2)

        # Compute the current theta based on EE position
        squeeze_axis = eef_pose[:3] - self.rotation_center

        theta_ee = np.arctan2(
            eef_pose[2] - self.rotation_center[2],
            np.dot(
                eef_pose[:2] - self.rotation_center[:2],
                -self.wall_normal[:2],
            ),
        )
        theta_err = np.maximum(theta - theta_ee, 0.0)
        # update the shrink factor if squeezed too much
        # project the error onto the current squeeze axis
        tangent_axis = self.R_wall @ np.array([np.sin(theta_ee), 0.0, np.cos(theta_ee)])
        axial_err_norm = np.dot(osc_pos_err, squeeze_axis)
        b = self.default_shrink_factor * self.b + np.clip(
            (axial_err_norm - self.axial_err_target) * self.axial_error_gain,
            -self.max_axial_correction,
            self.max_axial_correction,
        )
        print("b correction", b)
        # print(
        #     "b correction",
        #     b - self.default_shrink_factor * self.b,
        # )
        x_diff, z_diff = math_utils.trammel_of_archimedes(self.a, b, theta)
        # Create a pi/2 arc that has p_WC as the rotation center
        target_pos = self.rotation_center.copy()
        # TODO: rotate the pivot plane as needed. Right now it's on XZ plane
        target_pos[0] -= x_diff
        target_pos -= np.maximum(t - 1.0, 0.0) * 0.2 * self.wall_normal

        target_pos[2] += z_diff
        self.last_osc_pos_target = target_pos.copy()
        # Compute the current X_WE
        dpose_gain = 1.0  # 5.0
        dpose = np.zeros(6)
        dpose[:3] = (
            target_pos - X_WE[:3, 3]
        ) * dpose_gain  # math_utils.X_to_state_euler(X_E_Et, "zxy")

        ee_rot_err = (
            self.initial_ee_rot_inv * Rotation.from_matrix(X_WE[:3, :3])
        ).as_rotvec() * 1.0
        ee_rot_err = np.clip(ee_rot_err, -0.2, 0.2)
        # dpose[3:6] = ee_rot_err
        # print(
        #     "tangent correction",
        #     np.clip(
        #         tangent_axis * theta_err * self.tangent_gain,
        #         -self.max_tangent_correction,
        #         self.max_tangent_correction,
        #     ),
        # )
        dpose[:3] += np.clip(
            tangent_axis * theta_err * self.tangent_gain,
            -self.max_tangent_correction,
            self.max_tangent_correction,
        )

        # Tangent direction correction
        # Note the tangent direction is defined as eef-eef_target

        # print(
        #     "dpose",
        #     dpose,
        #     "\n",
        # )
        # don't move in the y direction
        # X_EEf = X_WE[:3, :3].T @ self.X_WE_start[:3, :3].T
        # dpose[3:] = Rotation.from_matrix(X_EEf).as_rotvec()
        # dpose[1] = 0.0
        # Compute the derivative of the orientation error
        return dpose, gripper_action

    def is_success(self, state, current_step, env_idx):
        if not self.done_squeezing or not self.initialized_pivot:
            return False
        elif (current_step - self.pivot_start_step) > self.action_steps * 1.1:
            return False
        current_rot = Rotation.from_quat(
            state["cubeA_quat"][env_idx].detach().cpu().numpy()
            if isinstance(state["cubeA_quat"], torch.Tensor)
            else state["cubeA_quat"][env_idx]
        )
        rot_diff = self.initial_obj_rot_inv * current_rot
        if current_step % 30 == 0:
            print("rot diff", np.linalg.norm(Rotation.as_rotvec(rot_diff)))
        return self.done
        # # TODO Check for rotation on the wall axis vs other axes
        # return np.allclose(
        #     np.linalg.norm(Rotation.as_rotvec(rot_diff)), np.pi / 2.0, atol=0.05
        # )

    def clear(self):
        self.X_WE_start = None
        self.p_WC = None
        self.r_CE = None  # (X_WE_start[:3, 3] - p_WC)[:2] * shrink_factor
        self.r_CE_norm = None  # np.linalg.norm(self.r_CE)
        self.done_squeezing = False


class RelocateController(BaseController):
    """
    This controller relocates the robot end effector
    """

    default_arm_q = np.array(
        [
            0.09162008114028396,
            -0.19826458111314524,
            -0.01990020486871322,
            -2.4732269941140346,
            -0.01307073642274261,
            2.30396583422025,
            0.8480939705504309,
        ]
    )

    def __init__(
        self,
        q_waypoints=None,
        q_tol=0.05,
        **kwargs,
    ) -> None:
        super().__init__()
        self.arm_q_waypoints = (
            [self.default_arm_q] if q_waypoints is None else q_waypoints
        )
        self.q_tol = q_tol

    def compute_action(self, *args, **kwargs):
        current_robot_q = (
            kwargs["state"]["q"][kwargs["env_idx"]].detach().cpu().numpy()
            if isinstance(kwargs["state"]["q"], torch.Tensor)
            else kwargs["state"]["q"][kwargs["env_idx"]]
        )
        if kwargs["current_step"] == 0:
            self.arm_q_waypoints.append(kwargs["current_primitive"].goal_robot_q[:7])
            self.final_gripper_q_target = np.sum(
                kwargs["current_primitive"].goal_robot_q[-2:]
            )
            self.current_waypoint = 0
            print("arm q waypoints", self.arm_q_waypoints)

        if len(self.arm_q_waypoints[self.current_waypoint]) == 7:
            gripper_target = self.final_gripper_q_target
        else:
            gripper_target = np.sum(self.arm_q_waypoints[self.current_waypoint][-2:])
        # Don't move the gripper until the last waypoint
        # FIXME: This is wrong
        # gripper_q_target = (
        #     self.current_waypoint
        #     if self.current_waypoint == len(self.arm_q_waypoints) - 1
        #     else (current_robot_q[-2] + current_robot_q[-1])
        # )
        if (
            np.linalg.norm(
                current_robot_q[:7] - self.arm_q_waypoints[self.current_waypoint][:7],
            )
            < self.q_tol
        ):
            self.current_waypoint += 1
            self.current_waypoint = min(
                self.current_waypoint, len(self.arm_q_waypoints) - 1
            )
        return (
            self.arm_q_waypoints[self.current_waypoint][:7],
            gripper_target,
        )

    def is_success(self, state, current_step, env_idx):
        if (
            np.linalg.norm(
                state["q"][env_idx, :7].detach().cpu().numpy()
                - self.arm_q_waypoints[self.current_waypoint][:7],
            )
            < self.q_tol
        ) and self.current_waypoint == len(self.arm_q_waypoints) - 1:
            return True
        return False

    def clear(self):
        pass


class GraspController(BaseController):
    """
    Perform a top down grasp of the object
    """

    def __init__(
        self,
        approach_height=0.05,
        descend_steps=50,
        close_gripper_steps=35,
        lift_steps=25,
        release_steps=30,
        has_lift=False,
        p_WR=np.zeros(3),
        **kwargs,
    ) -> None:
        super().__init__()
        self.current_stage = 0
        self.current_stage_start_step = 0
        self.descend_steps = descend_steps
        self.backup_steps = 18
        self.close_gripper_steps = close_gripper_steps
        self.lift_steps = lift_steps
        self.release_steps = release_steps

        self.approach_height = approach_height
        self.shimmy_early_stop_z = -0.03
        self.descend_success_z = -0.01

        self.shimmy_period = 60
        self.shimmy_magnitude = 0.5
        self.shimmy_depth = 0.015
        self.min_shimmy_magnitude = 0.05
        self.max_shimmy_magnitude = 0.15
        self.max_shimmy_steps = 400
        self.has_lift = has_lift
        self.xy_servo_gain = 1.0
        self.max_xy_servo = 0.01
        self.min_gripper_gap = 0.015
        self.p_WR = p_WR
        assert self.approach_height > 0

    def compute_action(self, state, current_step, *args, **kwargs):
        eef_pos = (
            state["eef_pos"][0].detach().cpu().numpy()
            if isinstance(state["eef_pos"], torch.Tensor)
            else state["eef_pos"][0]
        )
        eef_quat = (
            state["eef_quat"][0].detach().cpu().numpy()
            if isinstance(state["eef_quat"], torch.Tensor)
            else state["eef_quat"][0]
        )
        object_pos = (
            state["cubeA_pos"][0].detach().cpu().numpy()
            if isinstance(state["cubeA_pos"], torch.Tensor)
            else state["cubeA_pos"][0]
        ) - self.p_WR
        if current_step == 0:
            self.eef_pos_init = eef_pos.copy()
            self.eef_quat_init = eef_quat.copy()
            self.current_stage = "descend"
            self.current_stage_start_step = current_step
            print("OSC descending from height", self.eef_pos_init[2])
            self.shimmy_axis = Rotation.from_quat(eef_quat).as_matrix() @ np.array(
                [1.0, 0.0, 0.0]
            )
            self.grasp_target_z = (
                self.eef_pos_init[2] + self.descend_success_z - self.approach_height
            )

        action = np.zeros(7)
        if self.current_stage == "descend":
            # blind OSC descend
            pos_diff = object_pos - eef_pos
            if np.linalg.norm(pos_diff[:2]) > 0.03:
                print("Warning: large object position error", pos_diff)
            action[2] = -0.005
            action[:2] = pos_diff[:2] * self.xy_servo_gain
            xy_servo_norm = np.linalg.norm(action[:2])
            if xy_servo_norm > self.max_xy_servo:
                action[:2] = action[:2] * self.max_xy_servo / xy_servo_norm
            # finger_pointing_axis = Rotation.from_quat(eef_quat).as_matrix() @ np.array(
            #     [0.0, 0.0, 1.0]
            # )
            # print("finger_pointing_axis", finger_pointing_axis)
            # ray_to_object = object_pos - eef_pos
            # print("ray_to_object", ray_to_object)
            # rot_target = np.cross(
            #     finger_pointing_axis, ray_to_object / np.linalg.norm(ray_to_object)
            # )
            # rot_axis = np.cross(finger_pointing_axis, [0, 0, 1])
            # rot_diff = self.initial_obj_rot_inv * current_rot
            # print("rot_diff", rot_axis)
            # action[3:6] = -rot_axis * 2 if np.linalg.norm(rot_axis) > 0.0 else 0.0

            # action[0] = 0.00
            action[-1] = 1.0
            # If reach desired depth, just close gripper. Otherwise,
            # lift a bit and shimmy
            if current_step - self.current_stage_start_step > self.descend_steps:
                print(
                    "eef_pos[2] < self.grasp_target_z", eef_pos[2], self.grasp_target_z
                )
                if eef_pos[2] < self.grasp_target_z:
                    print("Reached desired depth")
                    # completed descend
                    self.current_stage_start_step = current_step
                    self.current_stage = "close_gripper"
                else:
                    self.current_stage = "shimmy"
                    self.current_stage_start_step = current_step
                    # Recompute grasp depth based on the current depth, since shimmy kicks in if the gripper is stuck on its way down
                    self.grasp_target_z = eef_pos[2] + self.shimmy_early_stop_z
                    self.shimmy_start_height = eef_pos[2]
                    print("Shimmy start height", self.shimmy_start_height)
                    print("shimmy depth", self.shimmy_depth)

        elif self.current_stage == "close_gripper":
            # Close gripper for 30 steps
            action[-1] = 0.0
            if current_step - self.current_stage_start_step > self.close_gripper_steps:
                self.current_stage_start_step = current_step
                self.current_stage = "lift" if self.has_lift else "done"

        elif self.current_stage == "lift":
            action[2] = 0.03
            action[0] = -0.02
            action[-1] = 0.0
            if current_step - self.current_stage_start_step > self.lift_steps:
                self.current_stage = "done"

        elif self.current_stage == "backup":
            raise DeprecationWarning
            # want to keep the gripper on the XY plane
            target_orn_mat = np.eye(3)
            target_orn_mat[:3, 0] = self.shimmy_axis
            target_orn_mat[:3, 2] = np.array([0.0, 0.0, 1.0])
            target_orn_mat[:3, 1] = np.cross(
                target_orn_mat[:3, 2], target_orn_mat[:3, 0]
            )
            action[2] = 0.01
            action[-1] = 1.0
            if current_step - self.current_stage_start_step > self.backup_steps:
                self.current_stage = "shimmy"
                self.current_stage_start_step = current_step
                # compute the shimmy axis which is along the x axis of the ee
        elif self.current_stage == "shimmy":
            # shimmy
            action[-1] = 1.0
            if eef_pos[2] > self.shimmy_start_height - self.shimmy_depth:
                action[2] = -0.003
                shimmy_magnitude = (
                    self.shimmy_magnitude
                    * (current_step - self.current_stage_start_step)
                    / self.max_shimmy_steps
                ) + self.min_shimmy_magnitude
                shimmy_magnitude = min(shimmy_magnitude, self.max_shimmy_magnitude)
                action[3:6] = shimmy_magnitude * (
                    np.cos(2.0 * np.pi * current_step / self.shimmy_period)
                    * self.shimmy_axis
                )

            else:
                print("pure descending")
                action[2] = -0.005
            # shimmy_translation = np.array([-self.shimmy_axis[1], self.shimmy_axis[0]])
            # action[:2] = 0.01 * (
            #     np.cos(2.0 * np.pi * current_step / self.shimmy_period)
            #     * shimmy_translation
            # )
            # action[4] = self.shimmy_magnitude * (
            #     np.sin(2.0 * np.pi * current_step / self.shimmy_period)
            # )

            if (
                (eef_pos[2] < self.grasp_target_z)
                or current_step - self.current_stage_start_step > self.max_shimmy_steps
            ):
                # completed descend
                print(
                    "ending shimmy, timeout",
                    current_step - self.current_stage_start_step
                    > self.max_shimmy_steps,
                )
                print("eef pos and grasp target z", eef_pos[2], self.grasp_target_z)
                self.current_stage_start_step = current_step
                self.current_stage = "close_gripper"
        action[:6] = np.clip(action[:6], -osc_max_action, osc_max_action)
        print("self.current_stage", self.current_stage)
        return action[:6], action[-1] * 0.08

    def is_success(self, *args, **kwargs):
        state = kwargs["state"]
        env_idx = kwargs["env_idx"]
        gripper_gap = (
            np.sum(state["q"][env_idx, 7:9].detach().cpu().numpy())
            if isinstance(state["q"], torch.Tensor)
            else np.sum(state["q"][env_idx][7:])
        )
        # Disable gripper gap check due to bug of gripper state not properly updated
        return self.current_stage == "done"  # and gripper_gap > self.min_gripper_gap

    def clear(self):
        self.current_stage = 0
        self.current_stage_start_step = 0


class PullController(BaseController):
    def __init__(self) -> None:
        super().__init__()
        self.max_pull_magnitude = 0.04
        self.goal_pos_tol = 0.04
        self.withdraw_steps = 25
        self.pull_gain = 5.0
        self.backup_steps = 10
        self.min_pull_magnitude = 0.02
        self.max_action = 0.04
        self.search_steps = 15

    def compute_action(self, state, current_step, env_idx, *args, **kwargs):
        eef_pos = (
            state["eef_pos"][env_idx].detach().cpu().numpy()
            if isinstance(state["eef_pos"], torch.Tensor)
            else state["eef_pos"][env_idx]
        )
        eef_quat = (
            state["eef_quat"][env_idx].detach().cpu().numpy()
            if isinstance(state["eef_quat"], torch.Tensor)
            else state["eef_quat"][env_idx]
        )
        current_obj_pos = (
            state["cubeA_pos"][env_idx].detach().cpu().numpy()
            if isinstance(state["cubeA_pos"], torch.Tensor)
            else state["cubeA_pos"][env_idx]
        )
        target_obj_pos = (
            state["cubeB_pos"][env_idx].detach().cpu().numpy()
            if isinstance(state["cubeB_pos"], torch.Tensor)
            else state["cubeB_pos"][env_idx]
        )
        gripper_action = 0.0
        # gripper_action = (
        #     np.sum(state["q"][env_idx, 7:9].detach().cpu().numpy())
        #     if isinstance(state["q"], torch.Tensor)
        #     else np.sum(state["q"][env_idx, 7:9])
        # )

        if current_step == 0:
            # Initialize the downward push
            self.state = "descend"
            self.osc_depth_target = eef_pos[2]
            self.eef_pre_descend = eef_pos.copy()
            self.num_search = 0
            self.search_magnitude = 0.01
        dpose = np.zeros(6)
        print("Current state:", self.state)
        if self.state == "descend":
            # Push down
            self.osc_depth_target -= 0.005
            dpose[2] = self.osc_depth_target - eef_pos[2]
            if np.linalg.norm(dpose[2]) > 0.03:  # to tune
                self.state = "pull"
                self.gripper_rot_inv = Rotation.from_quat(eef_quat).inv()
            if eef_pos[2] < self.eef_pre_descend[2] - 0.06:
                self.state = "backup"
        elif self.state == "backup":
            dpose[:3] = (self.eef_pre_descend[:3] - eef_pos[:3]) * 0.1
            dpose[2] = 0.01
            if self.eef_pre_descend[2] <= eef_pos[2]:
                self.state = "search"
                self.current_stage_start_step = current_step
                self.osc_depth_target = self.eef_pre_descend[2]
                self.num_search += 1
        elif self.state == "search":
            # Search for the object
            if self.num_search == 1:
                dpose[0] = self.search_magnitude
            elif self.num_search == 2:
                dpose[1] = self.search_magnitude
            elif self.num_search == 3:
                dpose[0] = -self.search_magnitude
            elif self.num_search == 4:
                dpose[1] = -self.search_magnitude
            else:
                self.num_search = 1
                self.search_magnitude *= 1.5

            if current_step - self.current_stage_start_step > self.search_steps:
                self.state = "descend"

        elif self.state == "pull":
            # Push the end effector toward the object
            xy_pos_err = eef_pos[:2] - target_obj_pos[:2]
            if current_step % 30 == 0:
                print("xy_pos_err", xy_pos_err, np.linalg.norm(xy_pos_err))
            dpose[:2] = -(xy_pos_err)
            if np.linalg.norm(dpose[:2]) > self.max_pull_magnitude:
                dpose[:2] = (
                    self.max_pull_magnitude * dpose[:2] / np.linalg.norm(dpose[:2])
                )
            elif np.linalg.norm(dpose[:2]) < self.min_pull_magnitude:
                dpose[:2] = (
                    self.min_pull_magnitude * dpose[:2] / np.linalg.norm(dpose[:2])
                )
            dpose[2] = np.clip(self.osc_depth_target - eef_pos[2], -0.01, 0.01)

            current_rot = Rotation.from_quat(eef_quat)
            rot_diff = self.gripper_rot_inv * current_rot
            dpose[3:6] = np.clip(rot_diff.as_rotvec() * 2.0, -0.4, 0.4)

            if np.linalg.norm(xy_pos_err) < self.goal_pos_tol:
                self.state = "withdraw"
                self.withdraw_start_step = current_step
        elif self.state == "withdraw":
            dpose[2] = 0.01
            if current_step - self.withdraw_start_step > self.withdraw_steps:
                self.state = "done"
        dpose = np.clip(dpose, -self.max_action, self.max_action)
        return dpose, gripper_action

    def is_success(self, state, env_idx, *args, **kwargs):
        # Compare the difference between the target state and the
        return self.state == "done"

    def clear(self):
        self.state = False


class GoToQController(BaseController):
    def __init__(self, arm_q_target, gripper_q_target=None, action_steps=50) -> None:
        super().__init__()
        self.arm_q_target = arm_q_target
        self.gripper_q_target = gripper_q_target
        self.action_steps = action_steps

    def compute_action(self, state, current_step, env_idx, *args, **kwargs):
        if self.gripper_q_target is not None:
            gripper_q_target = self.gripper_q_target
        else:
            gripper_q_target = np.sum(state["q"][env_idx, 7:].detach().cpu().numpy())
        return self.arm_q_target, gripper_q_target

    def is_success(self, state, current_step, env_idx):
        franka_q = state["q"][env_idx, :7].detach().cpu().numpy()
        if np.linalg.norm(self.arm_q_target - franka_q) < 0.05:
            return True

    def clear(self):
        raise NotImplementedError


class GoToDefaultQController(GoToQController):
    """
    Test controller that
    """

    def __init__(self, gripper_q_target=None, action_steps=50) -> None:
        self.franka_default_q = np.array(
            [
                0.09162008114028396,
                -0.19826458111314524,
                -0.01990020486871322,
                -2.4732269941140346,
                -0.01307073642274261,
                2.30396583422025,
                0.8480939705504309,
            ]
        )
        super().__init__(
            arm_q_target=self.franka_default_q,
            gripper_q_target=gripper_q_target,
            action_steps=action_steps,
        )


class GripperCommandController(BaseController):
    """
    Move gripper while fixing the arm q
    """

    def __init__(self, gripper_q_target, action_steps=25) -> None:
        super().__init__()
        self.gripper_q_target = gripper_q_target
        self.action_steps = action_steps

    def compute_action(self, *args, **kwargs):
        q = kwargs["state"]["q"][kwargs["env_idx"]].detach().cpu().numpy()
        return np.zeros(6), self.gripper_q_target

    def is_success(self, *args, **kwargs):
        return kwargs["current_step"] > self.action_steps

    def clear(self):
        pass


class OSCMoveController(BaseController):
    def __init__(self, default_action, action_steps=30) -> None:
        super().__init__()
        self.action_steps = action_steps
        self.default_action = default_action

    def compute_action(
        self, state, current_step, env_idx, action=None, *args, **kwargs
    ):
        if action is None:
            action = self.default_action
        return action[:6], (
            np.sum(state["q"][env_idx, 7:].detach().cpu().numpy())
            if len(action) == 6
            else np.sum(action[6:])
        )

    def is_success(self, *args, **kwargs):
        return kwargs["current_step"] > self.action_steps

    def clear(self, *args, **kwargs):
        pass
