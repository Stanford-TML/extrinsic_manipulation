import pybullet as p
import time
import numpy as np
import os
import pickle
import transforms3d

def init_env():
    table_pos = [0.0, 0.0, 1.0]
    table_thickness = 0.05
    table_xy_dims = [1, 1.2]

    # Create the floor (table) as a box
    table_half_extents = [
        table_xy_dims[0] / 2,
        table_xy_dims[1] / 2,
        table_thickness / 2,
    ]  # Size of the table (x, y, z)
    tableId = p.createCollisionShape(p.GEOM_BOX, halfExtents=table_half_extents)
    p.createMultiBody(baseCollisionShapeIndex=tableId, basePosition=table_pos)

    # Create table stand asset
    table_stand_dim = [0.2, 0.2, 0.013]
    table_stand_pos = [
        -0.5,
        0.0,
        1.0315
    ]

    table_stand_half_extents = [
        table_stand_dim[0] / 2,
        table_stand_dim[1] / 2,
        table_stand_dim[2] / 2,
    ]  # Size of the table (x, y, z)
    tableStandId = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=table_stand_half_extents
    )
    p.createMultiBody(
        baseCollisionShapeIndex=tableStandId, basePosition=table_stand_pos
    )

    # Create the wall as a box
    wall_size = [0.045, 1.015, 1]
    wall_half_extents = [s / 2 for s in wall_size]
    wall_pos = [
        0.4225,
        0.0,
        1.075
    ]

    wall_bbox = [
        wall_pos[0] - wall_size[0] / 2,
        wall_pos[0] + wall_size[0] / 2,
        wall_pos[1] - wall_size[1] / 2,
        wall_pos[1] + wall_size[1] / 2,
        wall_pos[2] - wall_size[2] / 2,
        wall_pos[2] + wall_size[2] / 2,
    ]
    table_bbox = [
        table_pos[0] - table_xy_dims[0] / 2,
        table_pos[0] + table_xy_dims[0] / 2,
        table_pos[1] - table_xy_dims[1] / 2,
        table_pos[1] + table_xy_dims[1] / 2,
        table_pos[2] - table_thickness / 2,
        table_pos[2] + table_thickness / 2,
    ]

    wallId = p.createCollisionShape(p.GEOM_BOX, halfExtents=wall_half_extents)
    p.createMultiBody(baseCollisionShapeIndex=wallId, basePosition=wall_pos)

    # Load the Franka robot with parallel gripper
    frankaStartPos = [
        -0.45,
        0,
        1.0 + table_thickness / 2 + table_stand_dim[2],
    ]  # Adjust the position as needed
    frankaId = p.loadURDF(
        "contactdemo/assets/franka_description/robots/franka_panda_gripper.urdf",
        frankaStartPos,
        useFixedBase=True,
    )

    # Get the number of DOFs for the Franka robot
    num_dofs = p.getNumJoints(frankaId)

    print("Number of DOFs for the Franka robot:", num_dofs)

    # Get the total number of joints for the Franka robot
    num_joints = p.getNumJoints(frankaId)

    # Get the number of joints with non-fixed type (active joints)
    num_active_joints = 0
    for joint_index in range(num_joints):
        joint_info = p.getJointInfo(frankaId, joint_index)
        joint_type = joint_info[2]
        if joint_type != p.JOINT_FIXED:
            num_active_joints += 1

    print("Number of active joints (DOFs) for the Franka robot:", num_active_joints)
    return {
        "table_id": tableId,
        "wall_id": wallId,
        "franka_id": frankaId,
        "wall_bbox": wall_bbox,
        "table_bbox": table_bbox,
    }


def set_franka_robot(franka_id, dofs):
    assert len(dofs) == 9
    # Get the total number of joints for the Franka robot
    num_joints = p.getNumJoints(franka_id)

    # Get the joint indices of the unfixed joints
    unfixed_joint_indices = []
    for joint_index in range(num_joints):
        joint_info = p.getJointInfo(franka_id, joint_index)
        joint_type = joint_info[2]
        if joint_type != p.JOINT_FIXED:
            unfixed_joint_indices.append(joint_index)

    # Set positions for the unfixed joints
    for i, joint_index in enumerate(unfixed_joint_indices):
        # p.setJointMotorControl2(franka_id, joint_index, p.POSITION_CONTROL, targetPosition=dofs[i])
        p.resetJointState(franka_id, joint_index, dofs[i])


def load_ycb_obj_by_id(ycb_id, state):
    # ycb_object_id = create_ycb_asset(ycb_id)
    ycb_object_id = create_ycb_asset(ycb_id)

    # Set the position and orientation of the object
    object_position = state[:3]
    # object_position[-1] -=0.1
    object_orientation = state[3:7]
    p.resetBasePositionAndOrientation(
        ycb_object_id, object_position, object_orientation
    )
    return ycb_object_id


if __name__ == "__main__":
    table_id, wall_id, franka_id, wall_boundary_x, table_boundary_z = init_env()
    ct = 0
    pos = np.array([1, 1, 1])
    rot = transforms3d.euler.euler2quat(0, 0, 0)
    load_ycb_obj_by_id(3, np.concatenate([pos, rot]))
    while True:
        p.stepSimulation()
        time.sleep(1.0 / 24.0)
        ct = ct + 1

    # Disconnect from the physics server
    p.disconnect()
