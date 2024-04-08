"""
Functions for computing the robot configuration and object pose for primitives,
given the initial object pose guess and the scene setup (via franka_station).

For a primitive sequence [P1, P2, P3, ...], the goal object state of Pi is
computed by the contact computation function of Pi+1. After reaching the goal of Pi,
the robot is relocated to the joint configuration computed by the contact computation
function of Pi+1... and so on.
"""

import contactdemo.lib.utils.math_utils as math_utils
import contactdemo.lib.drake.franka_table_station as franka_table_station
from isaacgymenvs.primitives.primitive_types import PrimitiveType, PrimitivePlan
import isaacgymenvs.primitives.primitive_controllers as primitive_controllers
import scipy.spatial.transform.rotation as R
import pydrake.solvers as mp
from pydrake.math import RotationMatrix
import numpy as np
import time


def compute_nearest_collision_free_object_pose_WB(
    franka_station,
    pose_WB_guess,
    p_WB_cost_scale=0.0001,  # this has to be small, otherwise the IK will fail. Hasn't work over 0.001
    visualize=False,
):
    ik = franka_station.construct_ik()
    # Encourage staying close to the initial guess
    X_WB_guess = math_utils.scipy_pose_to_X(pose_WB_guess)
    # if p_WB_cost_scale > 0:
    p_WB_cost = franka_station.add_obj_p_WB_cost(ik, X_WB_guess[:3, 3], p_WB_cost_scale)
    franka_station.add_obj_p_WB_in_AABB_constraint(
        ik,
        franka_station.world_lb,
        franka_station.world_ub,
    )
    q0 = np.zeros(16)
    # arm doesn't matter
    franka_station.get_franka_joint_angles(q0)[
        :7
    ] = franka_station.get_franka_default_arm_q()
    # gripper doesn't matter
    franka_station.get_franka_joint_angles(q0)[7:] = 0.04
    franka_station.get_quat_pos_WB(q0)[:] = math_utils.X_to_drake_pose_state(X_WB_guess)

    franka_station.visualize_q(q0)
    # print("visualizing guess")
    # time.sleep(2)

    prog = ik.get_mutable_prog()
    prog.SetInitialGuess(ik.q(), q0)
    result = mp.Solve(prog)
    if not result.is_success():
        raise AssertionError("Can't find collision free object pose")
    q_sol = result.GetSolution(ik.q())
    if visualize:
        franka_station.visualize_q(q_sol)
        print("visualizing solution 1")
        time.sleep(2)
    franka_station.add_object_pair_minimum_distance_constraint_to_ik(ik, 0.0)

    prog = ik.get_mutable_prog()
    prog.SetInitialGuess(ik.q(), q_sol)
    result = mp.Solve(prog)
    if not result.is_success():
        print("Can't find collision free object pose")
    q_sol = result.GetSolution(ik.q())
    if visualize:
        franka_station.visualize_q(q_sol)
        print("visualizing solution 2")
        time.sleep(10)

    return math_utils.drake_pose_state_to_scipy_pose_state(
        franka_station.get_quat_pos_WB(q_sol)
    )


def compute_object_X_WB_on_table(
    franka_station,
    X_WB_guess,
    xy_tol=0.01,
    z_tol=0.1,
    theta_bound=0.1,
):
    """
    Given a possibly noisy pose estimation of the object, find the closest
    pose where the object is sitting on the table with the provided orientation
    """
    # Now set up an IK to compute the approach q
    ik = franka_station.construct_ik()
    # Constrain the object position
    p_WB_lower = X_WB_guess[:3, 3].copy()
    p_WB_lower[:2] -= xy_tol
    p_WB_lower[2] -= z_tol
    p_WB_upper = X_WB_guess[:3, 3].copy()
    p_WB_upper[:2] += xy_tol
    p_WB_upper[2] += z_tol
    franka_station.add_obj_p_WB_in_AABB_constraint(ik, p_WB_lower, p_WB_upper)

    # constrain the object orientation
    franka_station.add_obj_R_WB_constraint(ik, X_WB_guess[:3, :3], theta_bound)
    franka_station.add_object_env_distance_constraint_to_ik(
        ik, 0.00, 0.005, franka_table_station.FrankaTableSceneObject.TABLE
    )
    q0 = np.zeros(16)
    # Initialize the robot to the default arm q
    franka_station.get_franka_joint_angles(q0)[:] = np.zeros(9)
    # Set the gripper to be open
    franka_station.get_franka_joint_angles(q0)[7:] = 0.04
    franka_station.get_quat_pos_WB(q0)[:] = math_utils.X_to_drake_pose_state(X_WB_guess)
    prog = ik.get_mutable_prog()
    prog.SetInitialGuess(ik.q(), q0)
    result = mp.Solve(prog)
    if not result.is_success():
        print("pose filtering IK failed")
        return X_WB_guess
    q_sol = result.GetSolution(ik.q())

    return math_utils.drake_pose_state_to_X(franka_station.get_quat_pos_WB(q_sol))


def compute_grasp_approach_q_given_X_WB(
    franka_station,
    X_WB,
    arm_q_guess,
    max_obj_width=0.085,
    approach_height=0.05,
):
    """
    Given the object's pose, compute an approach q and grasp q
    """
    # First validate the object pose is actually in contact with the table
    X_WB = compute_object_X_WB_on_table(franka_station, X_WB)
    # First set the franka station to the provided object pose
    # Use Open3D to compute the object's oriented bounding box
    # Returns a sequence of 4 robot q's, representing
    # 1. Approach q
    # 2. Grasp q
    # 3. Grasp q (closed gripper)
    # 4. Lift q
    obj_bbox_vert_W = np.zeros((8, 3))
    for idx, vert in enumerate(franka_station.obj_o3d_bbox_vertices):
        obj_bbox_vert_W[idx] = X_WB[:3, :3] @ vert + X_WB[:3, 3]
    top4_vertices = obj_bbox_vert_W[obj_bbox_vert_W[:, 2].argsort()][-4:]
    top_plane_center = np.average(top4_vertices, axis=0)
    # Compute the distance between the top 4 vertices
    # We also assume the 4 points are coplanar
    dist_to_firs_top_vert = np.linalg.norm(
        top4_vertices[1:, :2] - top4_vertices[0, :2], axis=1
    )
    nearest_vert = top4_vertices[np.argmin(dist_to_firs_top_vert) + 1]
    # The distance between these the two nearest vertices is the gripper width.
    # It also represent the short axis of the object's project on the xy plane
    planar_short_axis_length = np.min(dist_to_firs_top_vert)
    planar_short_axis_dir = (nearest_vert - top4_vertices[0])[:2]
    planar_short_axis_dir /= np.linalg.norm(planar_short_axis_dir)
    # Short axis is aligned with hand y
    # It should always carry negative y
    hand_planar_y_dir = planar_short_axis_dir.copy()
    if hand_planar_y_dir[1] > 0:
        hand_planar_y_dir *= -1
    hand_planar_x_dir = np.array([-hand_planar_y_dir[1], hand_planar_y_dir[0]])
    assert hand_planar_x_dir[0] >= -1e-3
    # Note that the franka frame has +z pointing in the finger extension direction
    if planar_short_axis_length > max_obj_width:
        raise ValueError("The object is too large for the gripper")
    # In Drake, x in the hand frame is the direction of the

    R_WH = -np.eye(3)
    R_WH[:2, 0] = hand_planar_x_dir
    R_WH[:2, 1] = hand_planar_y_dir
    print("R_WH", R_WH)

    # Now set up an IK to compute the approach q
    ik = franka_station.construct_ik()
    # The desired grasp pose has the hand's x-axis aligned with the long axis of the object
    fingertip_center = top_plane_center + np.array([0, 0, approach_height])
    franka_station.add_fingertip_center_to_p_W_constraint_to_ik(
        ik, fingertip_center - 0.01, fingertip_center + 0.01
    )
    # Move the object far away to facilitate IK solve
    object_ootw_pose = np.array([0, 0, 0, 1, 2, 0, 1])
    franka_station.add_obj_q_constraint(ik, object_ootw_pose, tol=5e-2)
    franka_station.add_hand_R_WH_constraint(ik, R_WH, 0.2)

    franka_station.add_arm_q_quadratic_cost(ik, arm_q_guess)
    q0 = np.zeros(16)
    # Initialize the robot to the default arm q
    franka_station.get_franka_joint_angles(q0)[:7] = arm_q_guess
    # Set the gripper to be open
    franka_station.get_franka_joint_angles(q0)[7:] = 0.04
    franka_station.get_quat_pos_WB(q0)[:] = (
        math_utils.scipy_pose_state_to_drake_pose_state(object_ootw_pose)
    )

    # DEBUG: visualize the scene
    prog = ik.get_mutable_prog()
    prog.SetInitialGuess(ik.q(), q0)
    result = mp.Solve(prog)
    q_sol1 = result.GetSolution(ik.q())
    approach_q = franka_station.get_franka_joint_angles(q_sol1)
    approach_q[7:] = 0.04
    franka_station.visualize_q(q_sol1)
    if not result.is_success():
        print("Grasp approach IK failed")
        breakpoint()
    return approach_q


def compute_pull_approach_q_given_X_WB(
    franka_station,
    X_WB,
    arm_q_guess,
    max_obj_width=0.085,
    approach_height=0.02,
):
    """
    Given the object's pose, compute an approach q and grasp q
    """
    # First validate the object pose is actually in contact with the table
    X_WB = compute_object_X_WB_on_table(franka_station, X_WB)
    # First set the franka station to the provided object pose
    # Use Open3D to compute the object's oriented bounding box
    # Returns a sequence of 4 robot q's, representing
    # 1. Approach q
    # 2. Grasp q
    # 3. Grasp q (closed gripper)
    # 4. Lift q
    obj_bbox_vert_W = np.zeros((8, 3))
    for idx, vert in enumerate(franka_station.obj_o3d_bbox_vertices):
        obj_bbox_vert_W[idx] = X_WB[:3, :3] @ vert + X_WB[:3, 3]
    top4_vertices = obj_bbox_vert_W[obj_bbox_vert_W[:, 2].argsort()][-4:]
    top_plane_center = np.average(top4_vertices, axis=0)
    # Compute the distance between the top 4 vertices
    # We also assume the 4 points are coplanar
    # Now set up an IK to compute the approach q
    ik = franka_station.construct_ik()
    # The desired grasp pose has the hand's x-axis aligned with the long axis of the object
    fingertip_center = np.array(
        [X_WB[0, 3], X_WB[1, 3], top_plane_center[2] + approach_height]
    )
    franka_station.add_fingertip_center_to_p_W_constraint_to_ik(
        ik, fingertip_center - 0.01, fingertip_center + 0.01
    )
    # Move the object far away to facilitate IK solve
    object_ootw_pose = np.array([0, 0, 0, 1, 2, 0, 1])
    franka_station.add_obj_q_constraint(ik, object_ootw_pose, tol=5e-2)
    franka_station.add_arm_q_quadratic_cost(ik, arm_q_guess)
    q0 = np.zeros(16)
    # Initialize the robot to the default arm q
    franka_station.get_franka_joint_angles(q0)[:7] = arm_q_guess
    franka_station.get_quat_pos_WB(q0)[:] = (
        math_utils.scipy_pose_state_to_drake_pose_state(object_ootw_pose)
    )

    # DEBUG: visualize the scene
    prog = ik.get_mutable_prog()
    prog.SetInitialGuess(ik.q(), q0)
    result = mp.Solve(prog)
    q_sol1 = result.GetSolution(ik.q())
    approach_q = franka_station.get_franka_joint_angles(q_sol1)
    approach_q[7:] = 0.0
    franka_station.visualize_q(q_sol1)
    if not result.is_success():
        print("Pull approach IK failed")
        breakpoint()
    return approach_q


def compute_initial_q_given_initial_X_WB_for_pivot(
    franka_station,
    X_WB=None,
    q_guess=None,
    obj_pose_tol=5e-3,
):
    if X_WB is None:
        assert q_guess is not None
    else:
        assert q_guess is None
        q_guess = np.zeros(16)
        # Franka default angles
        franka_station.get_franka_joint_angles(q_guess)[:] = np.array(
            [0, 0.1963, 0, -2.6180, 0, 2.9416, 0.7854, 0.035, 0.035]
        )
        # Set the initial guess for the object pose
        franka_station.get_quat_pos_WB(q_guess)[:] = math_utils.X_to_drake_pose_state(
            X_WB
        )
        # Ensure first X_WB is on the table
        X_WB = compute_object_X_WB_on_table(franka_station, X_WB)
    # Now fix the object pose and solve for franka pose
    ik = franka_station.construct_ik()
    # add object position constraint
    franka_station.add_obj_q_constraint(
        ik,
        franka_station.get_quat_pos_WB(q_guess),
        tol=obj_pose_tol,
    )

    # add collision constraint between finger and object
    robot_obj_dist_lower = 0.00
    robot_obj_dist_upper = 0.005

    # Constrain the fingers to be on the plane of the
    franka_station.add_fingertip_to_object_distance_constraint_to_ik(
        ik,
        franka_table_station.RobotFinger.LEFT,
        robot_obj_dist_lower,
        robot_obj_dist_upper,
    )
    franka_station.add_fingertip_to_object_distance_constraint_to_ik(
        ik,
        franka_table_station.RobotFinger.RIGHT,
        robot_obj_dist_lower,
        robot_obj_dist_upper,
    )
    franka_station.add_franka_finger_gap_constraint_to_ik(ik, min_gap=0.0, max_gap=0.04)

    X_WB_given = franka_station.get_X_WB(q_guess)
    # Compute the 8 vertices of the oriented bounding box in world frame when the object is at object_state
    box_points_hom = np.concatenate(
        [franka_station.obj_o3d_bbox_vertices, np.ones((8, 1))], axis=1
    )  # 8x4
    box_points_hom_W = (X_WB_given @ box_points_hom.T).T  # 4x8
    gaze_start = np.average(box_points_hom_W[:, :3], axis=0)
    # Gaze constraints
    franka_station.add_fingertip_in_wall_normal_gaze_cone_constraint(
        ik, franka_table_station.RobotFinger.LEFT, gaze_start, np.pi / 6
    )
    franka_station.add_fingertip_in_wall_normal_gaze_cone_constraint(
        ik, franka_table_station.RobotFinger.RIGHT, gaze_start, np.pi / 6
    )
    # Hand orientation constraint
    franka_station.add_franka_hand_orientaion_from_table_normal_cone_constraint(
        ik, np.pi / 6
    )
    # No backhand constraints
    franka_station.add_object_in_front_of_hand_gaze_constraint(ik, np.pi / 2)
    aabb_lower = np.array(
        [
            -np.infty,
            X_WB_given[1, 3] - 0.02,
            np.maximum(X_WB_given[2, 3] - 0.01, 1.03),  # slightly above table height
        ]  # Table height
    )
    aabb_upper = np.array([np.infty, X_WB_given[1, 3] + 0.02, X_WB_given[2, 3] + 0.01])
    # Require fingertips to be at the height of the object center
    franka_station.add_fingertip_world_position_constraint(
        ik, franka_table_station.RobotFinger.LEFT, aabb_lower, aabb_upper
    )
    franka_station.add_fingertip_world_position_constraint(
        ik, franka_table_station.RobotFinger.RIGHT, aabb_lower, aabb_upper
    )
    franka_station.add_fingertip_identical_gap_constraint(ik)
    # Solve the second IK
    prog = ik.get_mutable_prog()
    prog.SetInitialGuess(ik.q(), q_guess)
    result = mp.Solve(prog)
    return (result.GetSolution(ik.q()), result.is_success())


def compute_initial_pose_WB_for_pivot(
    franka_station, pose_WB_guess, max_pose_WB_x_deviation=0.15
):
    """
    Computes the goal pose for the primitive Pi, where its subsequent primitive Pi+1 is a pivot
    :franka_station: FrankaTableStation object that contains the scene setup (obstacle poses and object type)
    :pose_WB_guess: initial guess for the goal pose of the previous primitive in world frame, typically provided by the demonstration
    :return: the computed goal pose for primitive Pi and the initial joint configuration for Pi+1
    """
    # First, find an object state where the object is in contact with the wall & table
    # Then, fixing the object state,
    # find a end effector placement such that the fingertip is in contact with the object
    ik = franka_station.construct_ik()
    # add object position constraint
    X_WB_given = math_utils.pos_quat_to_X(pose_WB_guess[:3], pose_WB_guess[3:])
    # Compute the 8 vertices of the oriented bounding box in world frame when the object is at object_state
    box_points_hom = np.concatenate(
        [franka_station.obj_o3d_bbox_vertices, np.ones((8, 1))], axis=1
    )  # 8x4
    box_points_W = X_WB_given @ box_points_hom.T  # 4x8
    box_points_W = box_points_W[:3, :].T  # 8x3
    # Find the 2 points that are lowest and closest to the wall
    # first find the lowest 4
    lowest_4_idx = np.argsort(box_points_W[:, 2])[:4]
    # then find the closest 2 to the wall
    wall_points_sub_idx = np.argsort(
        franka_station.point_to_wall_xy_distance(box_points_W[lowest_4_idx, :])
    )[:2]
    wall_points_idx = lowest_4_idx[wall_points_sub_idx]
    # Compute the wall points in the object body frame
    wall_contact_points_B = box_points_hom[wall_points_idx, :3]
    # Constrain that these points are on the line against the wall
    for wall_contact_point_B in wall_contact_points_B:
        franka_station.add_obj_point_to_wall_bottom_dist_constraint_to_ik(
            ik, wall_contact_point_B, 0.0, 0.01
        )
    # Object BB should be above the table
    franka_station.add_obj_bounding_box_above_table_constraint(ik)

    # Add object orientation constraint
    franka_station.add_obj_R_WB_constraint(
        ik, X_WB_given[:3, :3], theta_bound=np.pi / 2.0
    )

    # The bottom edge of the oriented bounding box shoud be along the y-axis of wall_bottom_edge_frame
    # franka_station.add_obj_p_WB_constraint(ik, X_WB_given[:3, 3])
    # franka_station.add_obj_R_WB_constraint(ik, X_WB_given[:3, :3])

    # Constrain the object to be within the workspace
    world_lb = franka_station.world_lb.copy()
    world_lb[0] = pose_WB_guess[0] - max_pose_WB_x_deviation  # tolerance on x
    world_ub = franka_station.world_ub.copy()
    world_ub[0] = pose_WB_guess[0] + max_pose_WB_x_deviation  # tolerance on x

    franka_station.add_obj_p_WB_in_AABB_constraint(
        ik,
        franka_station.world_lb,
        franka_station.world_ub,
    )

    # object wall and table contacts
    franka_station.add_object_env_distance_constraint_to_ik(
        ik, 0.0, 0.002, franka_table_station.FrankaTableSceneObject.TABLE
    )
    franka_station.add_object_env_distance_constraint_to_ik(
        ik, 0.0, 0.002, franka_table_station.FrankaTableSceneObject.WALL
    )
    # Set up the initial guess
    q0 = np.zeros(16)
    # Initialize the robot to the default arm q
    franka_station.get_franka_joint_angles(q0)[:] = np.array(
        [0, 0.1963, 0, -2.6180, 0, 2.9416, 0.7854, 0.035, 0.035]
    )
    franka_station.get_p_WB(q0)[:] = pose_WB_guess[:3]
    franka_station.get_quat_WB(q0)[:] = (pose_WB_guess[3:])[
        [3, 0, 1, 2]
    ]  # scipy uses xyzw, drake uses wxyz

    prog = ik.get_mutable_prog()
    prog.SetInitialGuess(ik.q(), q0)
    result = mp.Solve(prog)
    return (result.GetSolution(ik.q()), result.is_success())


def compute_final_pose_WB_for_pivot(franka_station, initial_pose_WB):
    ik = franka_station.construct_ik()
    # add object position constraint
    initial_X_WB = math_utils.scipy_pose_to_X(initial_pose_WB)
    # Compute the 8 vertices of the oriented bounding box in world frame when the object is at object_state
    box_points_hom = np.concatenate(
        [franka_station.obj_o3d_bbox_vertices, np.ones((8, 1))], axis=1
    )  # 8x4
    box_points_W = initial_X_WB @ box_points_hom.T  # 4x8
    box_points_W = box_points_W[:3, :].T  # 8x3
    # Find the 2 points that are lowest and closest to the wall
    # first find the lowest 4
    lowest_4_idx = np.argsort(box_points_W[:, 2])[:4]
    upper_4_idx = np.argsort(box_points_W[:, 2])[-4:]

    # then find the closest 2 lower to the wall
    lower_wall_points_sub_idx = np.argsort(
        franka_station.point_to_wall_xy_distance(box_points_W[lowest_4_idx, :])
    )[:2]
    lower_wall_points_idx = lowest_4_idx[lower_wall_points_sub_idx]

    # find the closest 2 upper to the wall
    upper_wall_points_sub_idx = np.argsort(
        franka_station.point_to_wall_xy_distance(box_points_W[upper_4_idx, :])
    )[:2]
    upper_wall_points_idx = upper_4_idx[upper_wall_points_sub_idx]

    wall_point_pairs = []
    low_match_idx = np.argmin(
        np.linalg.norm(
            (
                box_points_W[upper_wall_points_idx[0]]
                - box_points_W[lower_wall_points_idx]
            )[:2]
        ),
    )
    wall_point_pairs.append(
        (upper_wall_points_idx[0], lower_wall_points_idx[low_match_idx])
    )
    wall_point_pairs.append(
        (upper_wall_points_idx[1], lower_wall_points_idx[1 - low_match_idx])
    )

    # Constrain that these points are on the line against the wall
    for upper_pt_idx, lower_pt_idx in wall_point_pairs:
        franka_station.add_obj_point_to_env_point_distance_constraint_to_ik(
            ik,
            franka_station.obj_o3d_bbox_vertices[upper_pt_idx],
            box_points_W[lower_pt_idx],
            0.0,
            0.01,
            "world",
        )
    # object wall and table contacts
    franka_station.add_object_env_distance_constraint_to_ik(
        ik, 0.0, 0.002, franka_table_station.FrankaTableSceneObject.TABLE
    )
    franka_station.add_object_env_distance_constraint_to_ik(
        ik, 0.0, 0.002, franka_table_station.FrankaTableSceneObject.WALL
    )
    # Set up the initial guess
    q0 = np.zeros(16)
    # Initialize the robot to the default arm q
    franka_station.get_franka_joint_angles(q0)[:] = np.array(
        [0, 0.1963, 0, -2.6180, 0, 2.9416, 0.7854, 0.035, 0.035]
    )

    # Obtain the initial pose guess by flipping the object pose by +90 deg along y
    rot_mat = R.Rotation.from_euler("y", np.pi / 2).as_matrix()
    X_WB_guess = initial_X_WB.copy()
    X_WB_guess[:3, :3] = rot_mat @ X_WB_guess[:3, :3]
    franka_station.get_quat_pos_WB(q0)[:] = math_utils.X_to_drake_pose_state(X_WB_guess)

    prog = ik.get_mutable_prog()
    prog.SetInitialGuess(ik.q(), q0)
    result = mp.Solve(prog)
    return math_utils.drake_pose_state_to_scipy_pose_state(
        franka_station.get_quat_pos_WB(result.GetSolution(ik.q()))
    )


def compute_q_given_push_start_goal_pose_WB(
    franka_station,
    X_WB_end,
    X_WB_start=None,
    q_guess=None,
    obj_pose_tol=2e-3,
):
    """
    Compute the robot q for the push primitive given the start and end object poses
    Note that the end pose is purely used as a heuristic for finger placement
    The robot hand should be opposite to the direction of the push.
    """
    # for now, just fill in default q and return
    franka_station.get_franka_joint_angles(q_guess)[:] = np.array(
        [0, 0.1963, 0, -2.6180, 0, 2.9416, 0.7854, 0.035, 0.035]
    )
    return q_guess, True
    # TODO
    if X_WB_start is None:
        assert q_guess is not None
    else:
        assert q_guess is None
        q_guess = np.zeros(16)
        # Franka default angles
        franka_station.get_franka_joint_angles(q_guess)[:] = np.array(
            [0, 0.1963, 0, -2.6180, 0, 2.9416, 0.7854, 0.035, 0.035]
        )
        # Set the initial guess for the object pose
        franka_station.get_p_WB(q_guess)[:] = X_WB_start[:3, 3]
        franka_station.get_quat_WB(q_guess)[:] = math_utils.rot_to_quat(
            X_WB_start[:3, :3]
        )
    # Now fix the object pose and solve for franka pose
    ik = franka_station.construct_ik()
    # add object position constraint
    franka_station.add_obj_q_constraint(
        ik,
        franka_station.get_quat_pos_WB(q_guess),
        tol=obj_pose_tol,
    )

    # add collision constraint between finger and object
    robot_obj_dist_lower = 0.00
    robot_obj_dist_upper = 0.005

    # Constrain the fingers to be on the plane of the
    franka_station.add_fingertip_to_object_distance_constraint_to_ik(
        ik,
        franka_table_station.RobotFinger.LEFT,
        robot_obj_dist_lower,
        robot_obj_dist_upper,
    )
    franka_station.add_fingertip_to_object_distance_constraint_to_ik(
        ik,
        franka_table_station.RobotFinger.RIGHT,
        robot_obj_dist_lower,
        robot_obj_dist_upper,
    )
    franka_station.add_franka_finger_gap_constraint_to_ik(ik, 0.02)

    X_WB_given = franka_station.get_X_WB(q_guess)
    # Compute the 8 vertices of the oriented bounding box in world frame when the object is at object_state
    box_points_hom = np.concatenate(
        [franka_station.obj_o3d_bbox_vertices, np.ones((8, 1))], axis=1
    )  # 8x4
    box_points_hom_W = (X_WB_given @ box_points_hom.T).T  # 4x8
    gaze_start = np.average(box_points_hom_W[:, :3], axis=0)
    # Gaze constraints
    franka_station.add_fingertip_in_wall_normal_gaze_cone_constraint(
        ik, franka_table_station.RobotFinger.LEFT, gaze_start, np.pi / 12
    )
    franka_station.add_fingertip_in_wall_normal_gaze_cone_constraint(
        ik, franka_table_station.RobotFinger.RIGHT, gaze_start, np.pi / 12
    )
    # Hand orientation constraint
    franka_station.add_franka_hand_orientaion_from_table_normal_cone_constraint(
        ik, np.pi / 12
    )
    # franka_station.add_object_in_front_of_hand_gaze_constraint(ik2, np.pi / 6)

    # Solve the second IK
    prog = ik.get_mutable_prog()
    prog.SetInitialGuess(ik.q(), q_guess)
    result = mp.Solve(prog)
    return (result.GetSolution(ik.q()), result.is_success())


def compute_next_initial_pose_WB_and_robot_q_given_next_primitive(
    franka_station,
    next_initial_pose_WB_guess,
    next_primitive_type,
    freeze_pose_WB=False,
):
    """
    Given the demonstration object pose, solve for the initial object pose to set as the previous primitive's goal
    Also solve for the initial robot q
    """
    if next_primitive_type == PrimitiveType.Pivot:
        # Compute the goal pose for the pivot primitive
        if not freeze_pose_WB:
            q_sol1, is_success1 = compute_initial_pose_WB_for_pivot(
                franka_station, next_initial_pose_WB_guess
            )
        else:
            q_sol1 = np.zeros(16)
            franka_station.get_quat_pos_WB(q_sol1)[:] = (
                math_utils.scipy_pose_state_to_drake_pose_state(
                    next_initial_pose_WB_guess
                )
            )
            q_sol1[:7] = franka_station.get_franka_default_arm_q()
            is_success1 = True
        q_sol2, is_success2 = compute_initial_q_given_initial_X_WB_for_pivot(
            franka_station, q_guess=q_sol1
        )
        # TODO: make this more lenient
        print("is_success", is_success1, is_success2)
        assert is_success1 and is_success2
        p_WB_sol = franka_station.get_p_WB(q_sol1)
        quat_WB_sol = math_utils.drake_quat_to_gym_quat(
            franka_station.get_quat_WB(q_sol1)
        )
        next_initial_pose_WB = np.concatenate([p_WB_sol, quat_WB_sol])
        next_initial_robot_q_sol = franka_station.get_franka_joint_angles(q_sol2)
        print("Computed pivot q", next_initial_robot_q_sol)
        return next_initial_pose_WB, next_initial_robot_q_sol

    elif next_primitive_type == PrimitiveType.Push:
        # Compute the goal pose for the push primitive
        if not freeze_pose_WB:
            object_pose_WB = compute_nearest_collision_free_object_pose_WB(
                franka_station, next_initial_pose_WB_guess
            )
        else:
            object_pose_WB = next_initial_pose_WB_guess
        default_q = np.zeros(9)
        default_q[:7] = franka_station.get_franka_default_arm_q()
        default_q[7:] = 0.04
        return object_pose_WB, default_q

    elif next_primitive_type == PrimitiveType.Grasp:
        approach_q = compute_grasp_approach_q_given_X_WB(
            franka_station,
            math_utils.scipy_pose_to_X(next_initial_pose_WB_guess),
            franka_station.get_franka_default_arm_q(),  # no initial guess
        )
        return next_initial_pose_WB_guess, approach_q
    elif next_primitive_type == PrimitiveType.Pull:
        approach_q = compute_pull_approach_q_given_X_WB(
            franka_station,
            math_utils.scipy_pose_to_X(next_initial_pose_WB_guess),
            franka_station.get_franka_default_arm_q(),  # no initial guess
        )
        return next_initial_pose_WB_guess, approach_q
    else:
        # TODO
        raise NotImplementedError


def compute_vicinity_waypoint_q_given_pose_WB_and_q(
    franka_station, pose_WB, robot_q, ray_length=0.07, step_size=0.005
):
    """
    Compute a "safe approach" q for relocation so the robot approximately approach
    the object in the surface normal using differential IK (approximated by mesh center - contact point) direction
    This is useful because the relocation is controlled in joint space, thus the EE may not
    approach in a straight line. Use the Jacobian to compute
    """
    current_q = np.zeros(16)
    franka_station.get_franka_joint_angles(current_q)[:] = robot_q
    franka_station.get_quat_pos_WB(current_q)[:] = (
        math_utils.scipy_pose_state_to_drake_pose_state(pose_WB)
    )
    approx_p_WH = np.zeros(3)  # hand pose
    for finger in [
        franka_table_station.RobotFinger.LEFT,
        franka_table_station.RobotFinger.RIGHT,
    ]:
        # Compute the contact point
        approx_p_WH += franka_station.get_fingertip_p_WF(
            current_q, finger
        ).translation()
    approx_p_WH /= 2
    # Compute the direction from the object center to the contact point
    ray_dir = approx_p_WH - pose_WB[:3]
    ray_dir /= np.linalg.norm(ray_dir)
    target_p_WH = approx_p_WH + ray_length * ray_dir
    # Re-solve the IK
    ik = franka_station.construct_ik()
    # add object position constraint
    franka_station.add_obj_q_constraint(
        ik,
        math_utils.scipy_pose_state_to_drake_pose_state(pose_WB),
    )
    # add fingertip position constraint
    franka_station.add_fingertip_center_to_p_W_constraint_to_ik(
        ik, target_p_WH - 0.01, target_p_WH + 0.01
    )
    prog = ik.get_mutable_prog()
    prog.SetInitialGuess(ik.q(), current_q)
    result = mp.Solve(prog)
    return (result.GetSolution(ik.q()), result.is_success())


def update_primitive_given_start_WB_change(
    primitive, franka_station=None, scene_cfg=None, object_file=None
):
    """
    Update the end pose of the primitive given the current robot q
    """
    if franka_station is None:
        assert scene_cfg is not None and object_file is not None
        franka_station = franka_table_station.FrankaTableStation(scene_cfg, object_file)
    assert primitive.start_pose_WB is not None
    if (
        primitive.primitive_type == PrimitiveType.Withdraw
        or primitive.primitive_type == PrimitiveType.Withdraw
    ):
        # Do nothing
        pass
    elif primitive.primitive_type == PrimitiveType.Pivot:
        # TODO
        if scene_cfg is None:
            # FIXME: this seems to make things worse when running on hardware
            primitive.goal_pose_WB = compute_final_pose_WB_for_pivot(
                franka_station, primitive.start_pose_WB
            )
    elif primitive.primitive_type == PrimitiveType.Grasp:
        primitive.goal_pose_WB = primitive.start_pose_WB

    elif primitive.primitive_type == PrimitiveType.Relocate:
        if primitive.next_primitive_type == PrimitiveType.Grasp:
            # Update the goal_robot_q given the current object pose
            primitive.goal_robot_q = compute_grasp_approach_q_given_X_WB(
                franka_station,
                math_utils.scipy_pose_to_X(primitive.start_pose_WB),
                primitive.goal_robot_q[:7],
                approach_height=0.05,
            )
        elif primitive.next_primitive_type == PrimitiveType.Pull:
            # Update the approach q
            primitive.goal_robot_q = compute_pull_approach_q_given_X_WB(
                franka_station,
                math_utils.scipy_pose_to_X(primitive.start_pose_WB),
                primitive.goal_robot_q[:7],
            )
    elif primitive.primitive_type == PrimitiveType.Push:
        # Check if there is collision with the final pose
        primitive.goal_pose_WB = compute_nearest_collision_free_object_pose_WB(
            franka_station,
            primitive.goal_pose_WB,
        )
    elif primitive.primitive_type == PrimitiveType.Pull:
        # Check if there is collision with the final pose
        primitive.goal_pose_WB = compute_nearest_collision_free_object_pose_WB(
            franka_station,
            primitive.goal_pose_WB,
        )
    else:
        raise NotImplementedError
    return primitive


def visualize_primitive_start_goal_pose(franka_station, primitive):
    franka_station_q = np.zeros(16)
    if primitive.start_robot_q is not None:
        franka_station.get_franka_joint_angles(franka_station_q)[
            :
        ] = primitive.start_robot_q
    franka_station.get_quat_pos_WB(franka_station_q)[:] = (
        math_utils.scipy_pose_state_to_drake_pose_state(primitive.start_pose_WB)
    )
    print("visualizing start pose")
    franka_station.visualize_q(franka_station_q)
    time.sleep(3)
    if primitive.goal_robot_q is not None:
        franka_station.get_franka_joint_angles(franka_station_q)[
            :
        ] = primitive.goal_robot_q
    franka_station.get_quat_pos_WB(franka_station_q)[:] = (
        math_utils.scipy_pose_state_to_drake_pose_state(primitive.goal_pose_WB)
    )
    print("visualizing end pose")
    franka_station.visualize_q(franka_station_q)
    time.sleep(3)


def retarget_primitive_sequence(
    original_primitive_seq,
    test_scene_cfg,
    object_file,
    observed_X_WB_init=None,
    visualize=False,
    disable_ik=False,
    override_max_num_step=600,
):
    """
    Run the initial solve for primitive sequence using the demonstration states
    :original_primitive_seq: list of PrimitivePlan
    :test_scene_cfg: Drake scene configuration, loaded from a file such as franka_table_scene.yaml
    :obj_file: path to the object file
    :observed_X_WB_init: the observed initial object pose, if None, use the start_pose_WB of the first primitive
    :disable_ik: disable IK computation and copy over the target states naively. Used for ablations
    """
    # First create franka_station
    if visualize:
        test_scene_cfg["visualize"] = True
    franka_station = franka_table_station.FrankaTableStation(
        test_scene_cfg, object_file
    )
    if observed_X_WB_init is None:
        observed_X_WB_init = math_utils.scipy_pose_to_X(
            original_primitive_seq[0].start_pose_WB
        )
    X_W_Wplus_list = []
    # Compute the transforms between the original waypoints
    # We neutralizes the initial orientation of the object
    R_raw_demo = (
        np.linalg.inv(
            math_utils.scipy_pose_to_X(original_primitive_seq[0].start_pose_WB)
        )
        @ observed_X_WB_init
    )
    R_raw_demo[:3, 3] = 0

    for i in range(len(original_primitive_seq)):
        X_W_Wplus_list.append(
            np.linalg.inv(
                math_utils.scipy_pose_to_X(original_primitive_seq[i].start_pose_WB)
            )
            @ math_utils.scipy_pose_to_X(original_primitive_seq[i].goal_pose_WB)
        )
    # Note that the goal of primitive i is identical to teh start of primitive i+1
    # Now update the primitives based on the observation
    original_primitive_seq[0].start_pose_WB = math_utils.X_to_pose(observed_X_WB_init)
    original_primitive_seq[0].goal_pose_WB = math_utils.X_to_pose(
        (observed_X_WB_init @ R_raw_demo.T) @ X_W_Wplus_list[0] @ R_raw_demo
    )
    # Clip the start and goal poses to the world size
    original_primitive_seq[0].goal_pose_WB[:3] = np.clip(
        original_primitive_seq[0].goal_pose_WB[:3],
        franka_station.world_lb,
        franka_station.world_ub,
    )
    if visualize:
        print(
            f"visualizing transformed primitive 0: {original_primitive_seq[0].primitive_type}"
        )
        visualize_primitive_start_goal_pose(franka_station, original_primitive_seq[0])
    for i in range(1, len(original_primitive_seq)):
        original_primitive_seq[i].start_pose_WB = original_primitive_seq[
            i - 1
        ].goal_pose_WB
        original_primitive_seq[i].goal_pose_WB = math_utils.X_to_pose(
            math_utils.scipy_pose_to_X(original_primitive_seq[i].start_pose_WB)
            @ R_raw_demo.T
            @ X_W_Wplus_list[i]
            @ R_raw_demo
        )
        # Clip the start and goal poses to the world size
        original_primitive_seq[i].goal_pose_WB[:3] = np.clip(
            original_primitive_seq[i].goal_pose_WB[:3],
            franka_station.world_lb,
            franka_station.world_ub,
        )
        if visualize:
            print(
                f"visualizing transformed primitive {i}: {original_primitive_seq[i].primitive_type}"
            )
            visualize_primitive_start_goal_pose(
                franka_station, original_primitive_seq[i]
            )
    # Then iterate through the primitive sequence and compute the goal object state for each primitive
    new_primitive_seq = []
    next_initial_pose_WB = None
    for i in range(len(original_primitive_seq)):
        current_primitive = original_primitive_seq[i]
        # assert current_primitive.goal_pose_WB is not None
        if current_primitive.goal_pose_WB is not None:
            current_primitive.goal_pose_WB = np.array(current_primitive.goal_pose_WB)
        # Override the start_pose_WB of this primitive with the one computed earlier, if it is available
        if next_initial_pose_WB is not None:
            current_primitive.start_pose_WB = next_initial_pose_WB  # Carried over from the previous loop, so it will contain the last initial pose
            # Update the primitive's end pose if the start is changed
            if not disable_ik:
                current_primitive = update_primitive_given_start_WB_change(
                    current_primitive, franka_station
                )

        # Set the primitive controllers
        if current_primitive.primitive_type == PrimitiveType.Pivot:
            current_primitive.controller = primitive_controllers.PivotController(
                action_steps=400, scene_cfg=test_scene_cfg
            )
            current_primitive.control_type = "osc"
        elif current_primitive.primitive_type == PrimitiveType.Push:
            current_primitive.control_type = "joint_imp"
            pass  # push uses learned policy
        elif current_primitive.primitive_type == PrimitiveType.Grasp:
            # No parameter is needed for grasp controller
            current_primitive.controller = primitive_controllers.GraspController()
            current_primitive.control_type = "osc"
            assert i == len(original_primitive_seq) - 1
        elif current_primitive.primitive_type == PrimitiveType.Pull:
            current_primitive.control_type = "osc"
            current_primitive.controller = primitive_controllers.PullController()
        else:
            raise ValueError("Unknown primitive type")

        if i == 0:
            if current_primitive.primitive_type == PrimitiveType.Pivot:
                pivot_initial_q_result, is_succ = (
                    compute_initial_q_given_initial_X_WB_for_pivot(
                        franka_station,
                        math_utils.scipy_pose_to_X(current_primitive.start_pose_WB),
                    )
                )
                assert is_succ
                vicinity_waypoint_result, is_succ = (
                    compute_vicinity_waypoint_q_given_pose_WB_and_q(
                        franka_station,
                        current_primitive.start_pose_WB,
                        franka_station.get_franka_joint_angles(pivot_initial_q_result),
                    )
                )
                pivot_start_robot_q = franka_station.get_franka_joint_angles(
                    pivot_initial_q_result
                )
                assert is_succ
                arm_q_waypoints = [
                    franka_station.get_franka_joint_angles(vicinity_waypoint_result)[
                        :7
                    ],
                    pivot_start_robot_q[:7],
                ]

                relocate_primitive = PrimitivePlan(
                    start_pose_WB=current_primitive.start_pose_WB,
                    goal_pose_WB=current_primitive.start_pose_WB,
                    goal_robot_q=pivot_start_robot_q,
                    primitive_type=PrimitiveType.Relocate,
                    max_num_steps=200,
                    controller=primitive_controllers.RelocateController(
                        arm_q_waypoints=arm_q_waypoints
                    ),
                    control_type="joint_target",
                )
                new_primitive_seq.append(relocate_primitive)
            elif current_primitive.primitive_type == PrimitiveType.Pull:
                pull_start_robot_q = compute_pull_approach_q_given_X_WB(
                    franka_station,
                    math_utils.scipy_pose_to_X(current_primitive.start_pose_WB),
                    franka_station.get_franka_default_arm_q(),  # no initial guess
                )
                vicinity_waypoint_result, is_succ = (
                    compute_vicinity_waypoint_q_given_pose_WB_and_q(
                        franka_station,
                        current_primitive.start_pose_WB,
                        franka_station.get_franka_joint_angles(pull_start_robot_q),
                    )
                )
                # Specify a 0 gripper q
                q_waypoints = [
                    franka_station.get_franka_joint_angles(vicinity_waypoint_result),
                    pull_start_robot_q,
                ]
                relocate_primitive = PrimitivePlan(
                    start_pose_WB=current_primitive.start_pose_WB,
                    goal_pose_WB=current_primitive.start_pose_WB,
                    goal_robot_q=pull_start_robot_q,
                    primitive_type=PrimitiveType.Relocate,
                    max_num_steps=200,
                    controller=primitive_controllers.RelocateController(
                        arm_q_waypoints=q_waypoints
                    ),
                    control_type="joint_target",
                )
                new_primitive_seq.append(relocate_primitive)

        if i == len(original_primitive_seq) - 1:
            # TODO: retarget the goal object state for the last primitive
            new_primitive_seq.append(current_primitive)
            # TODO: implement smart withdraw?
            if current_primitive.primitive_type == PrimitiveType.Grasp:
                # Add a relocate controller to the default
                relocate_primitive = PrimitivePlan(
                    start_pose_WB=current_primitive.goal_pose_WB,
                    goal_pose_WB=current_primitive.goal_pose_WB,
                    goal_robot_q=primitive_controllers.RelocateController.default_arm_q,
                    primitive_type=PrimitiveType.Relocate,
                    max_num_steps=200,
                    controller=primitive_controllers.GoToDefaultQController(
                        gripper_q_target=0.0
                    ),
                    control_type="joint_target",
                )
                new_primitive_seq.append(relocate_primitive)
                release_primitive = PrimitivePlan(
                    start_pose_WB=current_primitive.goal_pose_WB,
                    goal_pose_WB=current_primitive.goal_pose_WB,
                    goal_robot_q=primitive_controllers.RelocateController.default_arm_q,
                    primitive_type=PrimitiveType.Relocate,
                    max_num_steps=200,
                    controller=primitive_controllers.GripperCommandController(
                        gripper_q_target=0.08
                    ),
                    control_type="osc",
                )
                new_primitive_seq.append(release_primitive)
            elif current_primitive.primitive_type == PrimitiveType.Push:
                # Add a osc withdraw primitive to avoid hitting the object upon withdraw
                withdraw_primitive = PrimitivePlan(
                    start_pose_WB=current_primitive.goal_pose_WB,
                    goal_pose_WB=current_primitive.goal_pose_WB,
                    goal_robot_q=primitive_controllers.RelocateController.default_arm_q,
                    primitive_type=PrimitiveType.Withdraw,
                    max_num_steps=200,
                    controller=primitive_controllers.OSCMoveController(
                        default_action=np.array([-0.01, 0.0, 0.01, 0, 0, 0]),
                        action_steps=50,
                    ),
                    control_type="osc",
                )
                new_primitive_seq.append(withdraw_primitive)

            if visualize:
                print(
                    f"visualizing retargeted primitive {i}: {current_primitive.primitive_type}"
                )
                visualize_primitive_start_goal_pose(franka_station, current_primitive)

        else:
            next_primitive = original_primitive_seq[i + 1]
            # First compute a departure q for the current primitive
            assert next_primitive.primitive_type != PrimitiveType.Relocate
            # Update the goal of the current primitive based on the next primitive type
            next_initial_pose_WB, next_initial_robot_q = (
                compute_next_initial_pose_WB_and_robot_q_given_next_primitive(
                    franka_station,
                    current_primitive.goal_pose_WB,
                    next_primitive.primitive_type,
                    freeze_pose_WB=disable_ik,
                )
            )
            current_primitive.next_primitive_type = next_primitive.primitive_type
            # Update the goal of the current primitive based on the contact requirement of the next primitive
            current_primitive.goal_pose_WB = next_initial_pose_WB
            new_primitive_seq.append(current_primitive)
            if current_primitive.primitive_type == PrimitiveType.Push:
                # Add a osc withdraw primitive to avoid hitting the object upon withdraw
                withdraw_primitive = PrimitivePlan(
                    start_pose_WB=current_primitive.goal_pose_WB,
                    goal_pose_WB=current_primitive.goal_pose_WB,
                    goal_robot_q=primitive_controllers.RelocateController.default_arm_q,
                    primitive_type=PrimitiveType.Withdraw,
                    max_num_steps=200,
                    controller=primitive_controllers.OSCMoveController(
                        default_action=np.array([-0.01, 0.0, 0.01, 0, 0, 0, 0.02, 0.02])
                    ),
                    control_type="osc",
                )
                new_primitive_seq.append(withdraw_primitive)

            # Compute the waypoints for the relocate primitive
            result, is_succ = compute_vicinity_waypoint_q_given_pose_WB_and_q(
                franka_station, next_initial_pose_WB, next_initial_robot_q
            )
            if is_succ:
                default_robot_q = np.zeros(9)
                default_robot_q[:7] = (
                    primitive_controllers.RelocateController.default_arm_q
                )
                default_robot_q[7:] = 0.04
                arm_q_waypoints = [
                    default_robot_q,
                    franka_station.get_franka_joint_angles(result),
                ]

            else:
                raise AssertionError("Failed to find waypoint q")
            if visualize:
                print(f"visualizing retargeted primitive {i}")
                visualize_primitive_start_goal_pose(franka_station, current_primitive)
            # Add in a relocate primitive
            relocate_primitive = PrimitivePlan(
                start_pose_WB=current_primitive.goal_pose_WB,
                goal_pose_WB=current_primitive.goal_pose_WB,
                goal_robot_q=next_initial_robot_q,  # Relocate the robot to the desired next target
                primitive_type=PrimitiveType.Relocate,
                max_num_steps=200,
                controller=primitive_controllers.RelocateController(
                    q_waypoints=arm_q_waypoints,
                ),
                control_type="joint_target",
                next_primitive_type=next_primitive.primitive_type,
            )
            new_primitive_seq.append(relocate_primitive)
    return new_primitive_seq
