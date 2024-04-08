import numpy as np
import torch
from scipy.spatial.transform import Rotation


def state_to_X(state):
    x, y, z, qx, qy, qz, qw = state
    r = Rotation.from_quat([qx, qy, qz, qw])
    R = np.eye(4)
    R[:3, :3] = r.as_matrix()
    R[:3, 3] = [x, y, z]
    return R


def X_to_state(X):
    x, y, z = X[:3, 3]
    r = Rotation.from_matrix(X[:3, :3])
    qx, qy, qz, qw = r.as_quat()
    return np.array([x, y, z, qx, qy, qz, qw])


def X_to_state_euler(X, seq):
    # FIXME: is this correct?
    x, y, z = X[:3, 3]
    rot = Rotation.from_matrix(X[:3, :3])
    eul = rot.as_euler(seq)
    return np.array([x, y, z, *eul])


def quat_to_R(quat):
    r = Rotation.from_quat(quat)
    R = np.eye(4)
    return r.as_matrix()


def sample_uniform_random_translations(min, max, num_samples):
    shape = (num_samples, len(min))  # typically n*2 or n*3
    # (n*3) * ()
    return torch.rand(shape) * torch.tensor(max - min) + torch.tensor(min).view(1, -1)


def trammel_of_archimedes(a, b, t):
    """
    https://en.wikipedia.org/wiki/Trammel_of_Archimedes
    Compute the parameterized trajectory of a box rotating along the first quadrant
    from aligning with the x-axis to aligning with the y-axis
    The box has dimension axb, and the parameter t goes from 0 to pi/2
    At t=0, the point calculated has coordinates (b, a/2, 0)
    At t=pi/2, the point calculated has coordinates (a/2, b, 0)
    """
    assert a > 0
    assert b > 0
    # assert 0 <= t and t <= np.pi / 2
    y = a * np.cos(t)  # a to 0
    x = a * np.sin(t)  # 0 to a
    center = np.array([x, y]) / 2.0
    center += np.array([b * np.cos(t), b * np.sin(t)])
    return np.array([center[0], center[1]])
