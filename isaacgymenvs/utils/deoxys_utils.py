# For hardware interfacing via Deoxys
# https://github.com/UT-Austin-RPL/deoxys_control

import argparse

import numpy as np
import time
import os

current_path = os.path.dirname(os.path.realpath(__file__))
interface_cfg = "charmander.yml"
osc_controller_cfg_path = os.path.join(
    current_path, "../cfg/franka/osc-pose-controller.yml"
)
joint_controller_cfg_path = os.path.join(
    current_path, "../cfg/franka/joint-impedance-controller.yml"
)

try:
    from deoxys.franka_interface import FrankaInterface
    from deoxys import config_root
    from deoxys.utils.log_utils import get_deoxys_example_logger
    from deoxys.utils.config_utils import get_controller_config_from_file
    from deoxys.utils import transform_utils
    from deoxys.utils.motion_utils import interpolate_joint_positions
    from deoxys.experimental.motion_utils import position_only_gripper_move_by
except ImportError:
    print("Import deoxys failed. Unable to run on hardware.")


class DeoxysRobot:
    def __init__(self, use_visualizer=False) -> None:
        self.robot_interface = FrankaInterface(
            config_root + f"/{interface_cfg}", use_visualizer=use_visualizer
        )
        self.osc_controller_cfg = get_controller_config_from_file(
            osc_controller_cfg_path
        )
        self.osc_max_action = np.array(self.osc_controller_cfg["max_action"])
        self.joint_controller_cfg = get_controller_config_from_file(
            joint_controller_cfg_path
        )

        # u_arm, u_gripper = self.actions[:, :-1], self.actions[:, -1]
        self.logger = get_deoxys_example_logger()
        # Golden resetting joints
        self.default_arm_q = [
            0.09162008114028396,
            -0.19826458111314524,
            -0.01990020486871322,
            -2.4732269941140346,
            -0.01307073642274261,
            2.30396583422025,
            0.8480939705504309,
        ]
        self.default_gripper_q = [1.0]  # 0(close)~1(open)
        self.max_delta_q = np.array(self.joint_controller_cfg["max_delta_q"])

    def __del__(self):
        self.robot_interface.close()

    def set_q_to(self, q_target, arm_tol=5e-3, gripper_tol=3e-2):
        assert len(q_target) == 8
        # assert 0 <= q_target[7] <= 1
        arm_q_reached = False
        gripper_q_reached = False
        arm_err = np.inf
        gripper_err = np.inf
        while not arm_q_reached or not gripper_q_reached:
            if len(self.robot_interface._state_buffer) > 0:
                self.logger.debug(
                    f"Current Robot joint: {np.round(self.last_arm_q, 3)}"
                )
                self.logger.debug(
                    f"Desired Robot joint: {np.round(self.robot_interface.last_q_d, 3)}"
                )
                arm_err = np.max(
                    np.abs(
                        np.array(self.robot_interface._state_buffer[-1].q)
                        - np.array(q_target[:-1])
                    )
                )
                if arm_err < arm_tol:
                    arm_q_reached = True
            if self.robot_interface.last_gripper_q is not None:
                # Note that gripper q is from 0~1
                self.logger.debug(
                    f"Current gripper joint: {np.round(self.last_gripper_q, 3)}"
                )
                self.logger.debug(f"Desired gripper joint: {np.round(q_target[-1], 3)}")
                gripper_err = np.abs(self.last_gripper_q - q_target[-1])
                # if gripper_err < gripper_tol:
                gripper_q_reached = True
            if not arm_q_reached:
                self.robot_interface.control(
                    controller_type="JOINT_POSITION",
                    action=q_target,
                    controller_cfg=self.joint_controller_cfg,
                )
            if not gripper_q_reached:
                self.robot_interface.gripper_control(q_target[-1])

    def set_q_to_default(self):
        self.set_q_to(self.default_arm_q + self.default_gripper_q)

    @property
    def last_arm_q(self):
        return self.robot_interface.last_q

    @property
    def last_gripper_q(self):
        """Note that the gripper q has been mapped to 0(closed)~1(open)"""
        return self.robot_interface.last_gripper_q / 0.08

    @property
    def last_eef_pose(self):
        return self.robot_interface.last_eef_pose

    @property
    def last_eef_rot_and_pos(self):
        return self.robot_interface.last_eef_rot_and_pos

    @property
    def last_eef_quat_and_pos(self):
        return self.robot_interface.last_eef_quat_and_pos

    def move_up_to_default_q_to_target(self, up_height, final_q):
        """
        Move the end effector up, traverse the
        This is a hacky way to move the end effector to a new prescribed location e.g. contact points.
        A better implementation should be provided.
        """
        raise NotImplementedError
        # Move up
        position_only_gripper_move_by(self.robot_interface, [0, 0, up_height], 20)
        # Traverse to robot default q
        self.set_q_to_default()
        # Move down to q
        self.set_q_to(final_q)

    def osc_move_displacement(self, action, num_steps=1):
        """
        Move the end effector by the prescribed displacement using operational space control
        """
        assert len(action) == 7  # 3 for position, 3 for orientation, 1 for gripper
        if np.any(np.abs(action[:6]) > self.osc_max_action):
            self.logger.warn(
                f"OSC joint action too large, clipping. Original action {np.round(action, 3)}"
            )
            action[:6] = np.clip(action[:6], -self.osc_max_action, self.osc_max_action)
        for _ in range(num_steps):
            self.robot_interface.control(
                controller_type="OSC_POSE",
                action=action,
                controller_cfg=self.osc_controller_cfg,
            )

    def osc_move(
        self, robot_interface, controller_type, controller_cfg, target_pose, num_steps
    ):
        target_pos, target_quat = target_pose
        target_axis_angle = transform_utils.quat2axisangle(target_quat)
        current_rot, current_pos = robot_interface.last_eef_rot_and_pos
        raise AssertionError
        for _ in range(num_steps):
            current_pose = robot_interface.last_eef_pose
            current_pos = current_pose[:3, 3:]
            current_rot = current_pose[:3, :3]
            current_quat = transform_utils.mat2quat(current_rot)
            if np.dot(target_quat, current_quat) < 0.0:
                current_quat = -current_quat
            quat_diff = transform_utils.quat_distance(target_quat, current_quat)
            current_axis_angle = transform_utils.quat2axisangle(current_quat)
            axis_angle_diff = transform_utils.quat2axisangle(quat_diff)
            action_pos = (target_pos - current_pos).flatten() * 10
            action_axis_angle = axis_angle_diff.flatten() * 1
            action_pos = np.clip(action_pos, -1.0, 1.0)
            action_axis_angle = np.clip(action_axis_angle, -0.5, 0.5)

            action = action_pos.tolist() + action_axis_angle.tolist() + [-1.0]
            logger.info(f"Axis angle action {action_axis_angle.tolist()}")
            # print(np.round(action, 2))
            robot_interface.control(
                controller_type=controller_type,
                action=action,
                controller_cfg=controller_cfg,
            )
        return action

    def move_to_target_pose(
        self,
        robot_interface,
        controller_type,
        controller_cfg,
        target_delta_pose,
        num_steps,
        num_additional_steps,
        interpolation_method,
    ):
        raise NotImplementedError
        while robot_interface.state_buffer_size == 0:
            self.logger.warn("Robot state not received")
            time.sleep(0.5)

        target_delta_pos, target_delta_axis_angle = (
            target_delta_pose[:3],
            target_delta_pose[3:],
        )
        current_ee_pose = robot_interface.last_eef_pose
        current_pos = current_ee_pose[:3, 3:]
        current_rot = current_ee_pose[:3, :3]
        current_quat = transform_utils.mat2quat(current_rot)
        current_axis_angle = transform_utils.quat2axisangle(current_quat)

        target_pos = np.array(target_delta_pos).reshape(3, 1) + current_pos

        target_axis_angle = np.array(target_delta_axis_angle) + current_axis_angle

        self.logger.info(f"Before conversion {target_axis_angle}")
        target_quat = transform_utils.axisangle2quat(target_axis_angle)
        target_pose = target_pos.flatten().tolist() + target_quat.flatten().tolist()

        if np.dot(target_quat, current_quat) < 0.0:
            current_quat = -current_quat
        target_axis_angle = transform_utils.quat2axisangle(target_quat)
        self.logger.info(f"After conversion {target_axis_angle}")
        current_axis_angle = transform_utils.quat2axisangle(current_quat)

        start_pose = current_pos.flatten().tolist() + current_quat.flatten().tolist()

    # def joint_imp_action(self, q_target):
    #     assert len(q_target) == 8
    #     # Safety features
    #     while self.robot_interface.last_q is None:
    #         time.sleep(0.1)
    #     # Get the current robot state
    #     delta_q = self.robot_interface.last_q - q_target[:7]
    #     # check if any of delta_q is out of range
    #     if np.any(np.abs(delta_q) > self.max_delta_q):
    #         self.logger.warn(
    #             f"Arm joint target too far, using position control. Original target {np.round(q_target[:7], 3)}, current joint {np.round(self.robot_interface.last_q, 3)}"
    #         )
    #         self.set_q_to(q_target)
    #         return
    #     # # Clip delta q
    #     # delta_q = np.clip(
    #     #     delta_q,
    #     #     -self.max_delta_q,
    #     #     self.max_delta_q,
    #     # )
    #     # # Update the target
    #     # q_target[:7] = self.robot_interface.last_q - delta_q
    #     # q_arm, q_gripper = q_target[:-1], q_target[-1]
    #     self.robot_interface.control(
    #         controller_type="JOINT_IMPEDANCE",
    #         action=q_target,
    #         controller_cfg=self.joint_controller_cfg,
    #     )

    #     # self.robot_interface.gripper_control(q_gripper)

    def joint_imp_action(
        self,
        q_target,
        num_steps,
        num_additional_steps,
        interpolation_method,
    ):
        """
        Use joint impedance controller to move to a new joint position using interpolation.
        """
        assert len(q_target) == 8
        assert 0 <= q_target[7] <= 1
        # Check if the actions is withing the maximum step size
        current_q = self.robot_interface.last_q
        delta_q = q_target[:7] - current_q
        if np.any(np.abs(delta_q) > self.max_delta_q):
            self.logger.warn(
                f"Arm joint target too far, rejecting action. Original target {np.round(q_target[:7], 3)}, current joint {np.round(self.robot_interface.last_q, 3)}"
            )
            return self.robot_interface.last_q

        while True:
            # If the robot is already close to the target, skip
            if len(self.robot_interface._state_buffer) > 0:
                if (
                    np.max(np.abs(np.array(self.robot_interface._state_buffer[-1].q)))
                    < 1e-3
                ):
                    print(
                        len(self.robot_interface._state_buffer),
                        np.array(self.robot_interface._state_buffer[-1].q),
                    )
                    continue
                else:
                    break
        q_current = np.zeros(8)
        # FIXME: incorrect gripper action
        q_current[:7] = self.robot_interface.last_q
        # Note the q_target[7] is from 0 to 1
        q_current[7] = self.robot_interface.last_gripper_q

        # interpolate to get joint space path
        jpos_steps = interpolate_joint_positions(
            start_jpos=q_current,
            end_jpos=q_target,
            num_steps=num_steps,
            interpolation_method=interpolation_method,
        )
        # try to follow path
        controller_type = "JOINT_IMPEDANCE"
        for i, jpos_t in enumerate(jpos_steps):
            # Scale back the gripper angle
            jpos_t[7] /= 0.08
            # Send interpolated steps
            action = list(jpos_t)
            # print("step {}, action {}".format(i, np.round(action, 2)))
            self.robot_interface.control(
                controller_type=controller_type,
                action=action,
                controller_cfg=self.joint_controller_cfg,
            )
        for i in range(num_additional_steps):
            # Scale back q_target to 0~0.08
            action = list(q_target)
            action[7] /= 0.08
            # print("additional step {}, action {}".format(i, np.round(action, 2)))

            self.robot_interface.control(
                controller_type=controller_type,
                action=action,
                controller_cfg=self.joint_controller_cfg,
            )
        return self.robot_interface.last_q
