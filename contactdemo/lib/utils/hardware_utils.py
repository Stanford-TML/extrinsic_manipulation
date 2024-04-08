import numpy as np
from scipy.spatial.transform import Rotation as R


def compute_wall_yaw_offset(left_x, right_x, wall_width):
    """
    Given the x positions of the left and right ends of the wall,
    compute the position and orientation of the wall
    left and right x positions are relative to the robot (+x)
    """
    return np.arctan2(right_x - left_x, wall_width)


def update_cfg_wall_pos_orn(left_x, right_x, cfg):
    """
    Given the x positions of the left and right ends of the wall,
    compute the position and orientation of the wall and update
    left and right x positions are relative to the robot (+x)
    """
    # Compute the rotation of the wall
    yaw = compute_wall_yaw_offset(left_x, right_x, cfg["wall"]["wall_dim"][1])
    R_wall = R.from_euler("z", yaw).as_matrix()
    # Compute the position of the wall

    wall_center_shift = R_wall @ np.array([cfg["wall"]["wall_dim"][0] / 2.0, 0, 0])
    cfg["wall"]["wall_pos"][0] = (
        np.average([left_x, right_x]) + cfg["franka"]["franka_pos"][0]
    )
    cfg["wall"]["wall_pos"] += wall_center_shift
    cfg["wall"]["wall_rpy"][2] = yaw

    # Save the local variables for convenience
    cfg["wall"]["left_x"] = left_x
    cfg["wall"]["right_x"] = right_x

    return cfg
