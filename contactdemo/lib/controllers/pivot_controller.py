import numpy as np
import matplotlib.pyplot as plt


def plot_pivot_fn_output(
    pivot_trajectory_fn,
    X_WE_start=None,
    p_WC=None,
    fig=None,
    ax=None,
    num_pts=100,
):
    """
    Utility function to plot the pivot trajectory
    """
    ts = np.linspace(0, 1, num_pts)
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
    target_X_WEs = np.zeros((len(ts), 4, 4))
    for i, t in enumerate(ts):
        target_X_WEs[i] = pivot_trajectory_fn(t)
    target_p_WEs = target_X_WEs[:, :3, 3]
    # scatter the start and end
    fig, ax = plot_pivot_trajectory_data(target_p_WEs, fig, ax)
    if X_WE_start is not None:
        ax.scatter(
            X_WE_start[0, 3],
            X_WE_start[1, 3],
            X_WE_start[2, 3],
            color="red",
        )
    if p_WC is not None:
        ax.scatter(p_WC[0], p_WC[1], p_WC[2], color="green", marker="^")
    return fig, ax


def plot_pivot_trajectory_data(p_WE_data, fig=None, ax=None, colors=None):
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
    if len(p_WE_data.shape) == 2:
        ax.scatter(p_WE_data[:, 0], p_WE_data[:, 1], p_WE_data[:, 2], color=colors)
    else:
        for i in range(p_WE_data.shape[0]):
            ax.scatter(
                p_WE_data[i, :, 0],
                p_WE_data[i, :, 1],
                p_WE_data[i, :, 2],
                color=colors[i],
            )

    return fig, ax


class FrankaPivotController:
    def __init__(self, X_WE_start, p_WC, shrink_factor=0.97):
        # TODO: add robot to for feedback control
        self.pivot_target_fn = self.create_pivot_target_fn(
            X_WE_start, p_WC, shrink_factor
        )

    def create_pivot_target_fn(self, X_WE_start, p_WC, shrink_factor):
        """
        Create a pivot trajectory that is an arc
        :param X_WE_start: The initial pose of the end effector in world frame. The EE should
        be in contact with the object
        :param p_WC: The object-wall contact point in world frame. This is the center of rotation
        :
        """
        # Note that we do not consider the z component of r
        r_CE = (X_WE_start[:3, 3] - p_WC)[:2] * shrink_factor
        r_CE_norm = np.linalg.norm(r_CE)

        def pivot_trajectory(t):
            """
            Callable function that returns the OSC targets (in world frame) for pivoting
            If t=0, the target pose is X_WE_start
            """
            assert t >= 0.0 and t <= 1.0
            theta = np.pi / 2 * t
            # Create a pi/2 arc that has p_WC as the rotation center
            X_WE_target = X_WE_start.copy()
            X_WE_target[:3, 3] = p_WC
            X_WE_target[:2, 3] += r_CE * np.cos(theta)
            X_WE_target[2, 3] += r_CE_norm * np.sin(theta)  # z should change
            return X_WE_target

        return pivot_trajectory
