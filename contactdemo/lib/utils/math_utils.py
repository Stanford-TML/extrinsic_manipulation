from scipy.spatial.transform import Rotation as scipy_rot
import numpy as np


def X_to_pos_quat(X):
    return X[:3, 3], scipy_rot.from_matrix(np.array(X[:3, :3])).as_quat()


def X_to_pose(X):
    return np.concatenate(X_to_pos_quat(X))


def pos_quat_to_X(pos, quat):
    return np.block(
        [[scipy_rot.from_quat(quat).as_matrix(), pos.reshape(3, 1)], [0, 0, 0, 1]]
    )


def scipy_pose_to_X(pose):
    return np.block(
        [
            [scipy_rot.from_quat(pose[3:]).as_matrix(), pose[:3].reshape(3, 1)],
            [0, 0, 0, 1],
        ]
    )


def quat_diff(q1, q2):
    R1 = scipy_rot.from_quat(q1).as_matrix()
    R2 = scipy_rot.from_quat(q2).as_matrix()
    return scipy_rot.from_matrix(R2.T @ R1).as_quat()


def X_to_pos_axis_angle(X):
    return X[:3, 3], scipy_rot.from_matrix(np.array(X[:3, :3])).as_rotvec()


def X_to_drake_pose_state(X):
    """
    Convert a 4x4 matrix to drake pose convention, expressed as
    qw-qx-qy-qz-x-y-z
    """
    pos, quat = X_to_pos_quat(X)
    quat = quat[[3, 0, 1, 2]]  # scipy uses xyzw, drake uses wxyz
    return np.concatenate([quat, pos])


def drake_quat_to_gym_quat(q):
    return q[[1, 2, 3, 0]]


def drake_pose_state_to_X(q):
    """
    Convert a drake pose convention, expressed as
    qw-qx-qy-qz-x-y-z
    to a 4x4 matrix
    """
    quat = q[:4]
    quat = quat[[1, 2, 3, 0]]  # drake uses wxyz, scipy uses xyzw
    pos = q[4:]
    return pos_quat_to_X(pos, quat)


def scipy_pose_state_to_drake_pose_state(q):
    """
    Convert a scipy pose convention, expressed as
    x-y-z-qx-qy-qz-qw
    to a drake pose convention, expressed as
    qw-qx-qy-qz-x-y-z
    """
    quat = q[3:]
    quat = quat[[3, 0, 1, 2]]  # scipy uses xyzw, drake uses wxyz
    pos = q[:3]
    return np.concatenate([quat, pos])


def drake_pose_state_to_scipy_pose_state(q):
    """
    Convert a drake pose convention, expressed as
    qw-qx-qy-qz-x-y-z
    to a scipy pose convention, expressed as
    x-y-z-qx-qy-qz-qw
    """
    quat = q[:4]
    quat = quat[[1, 2, 3, 0]]  # drake uses wxyz, scipy uses xyzw
    pos = q[4:]
    return np.concatenate([pos, quat])
