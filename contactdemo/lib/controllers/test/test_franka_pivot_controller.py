import unittest
from contactdemo.lib.drake.state_estimator import StateEstimator
from contactdemo.lib.controllers.pivot_controller import *
import numpy as np
import matplotlib.pyplot as plt


class TestFrankaPivotController(unittest.TestCase):
    def setUp(self):
        # Set up any necessary dependencies or mocks
        # Create random start and end pose
        rs = np.random.RandomState(0)
        self.X_WE_start = np.eye(4)
        self.X_WE_start[:3, 3] = rs.randn(3)
        self.p_WC = rs.randn(3)
        # set p_WC to be at the same height as the start pose
        self.p_WC[2] = self.X_WE_start[2, 3]
        self.shrink_factor = 0.95
        self.franka_pivot_controller = FrankaPivotController(
            self.X_WE_start, self.p_WC, self.shrink_factor
        )

    def test_create_pivot_trajectory_fn(self):
        ts = np.linspace(0, 1, 100)
        # Set up any necessary test data or inputs
        dut = self.franka_pivot_controller._create_pivot_trajectory_fn(
            self.X_WE_start, self.p_WC, self.shrink_factor
        )
        target_X_WEs = np.zeros((len(ts), 4, 4))
        for i, t in enumerate(ts):
            target_X_WEs[i] = dut(t)
        # Assert the expected behavior or outcomes
        dist_from_p_WC = np.linalg.norm(target_X_WEs[:, :3, 3] - self.p_WC, axis=1)
        r_CE_expected = np.linalg.norm((self.p_WC - self.X_WE_start[:3, 3])[:2])
        self.assertTrue(
            np.allclose(
                dist_from_p_WC,
                r_CE_expected * self.shrink_factor,
            )
        )
        # Keep orientation
        self.assertTrue(np.allclose(target_X_WEs[:, :3, :3], self.X_WE_start[:3, :3]))
        dut2 = self.franka_pivot_controller._create_pivot_trajectory_fn(
            self.X_WE_start, self.p_WC, 1.0
        )
        # check start and end are equal to the input
        self.assertTrue(np.allclose(dut2(0), self.X_WE_start))
        self.assertTrue(
            np.allclose(np.linalg.norm(dut2(1)[:3, 3] - self.p_WC), r_CE_expected)
        )
        # Plot
        fig, ax = plot_pivot_fn_output(
            dut, self.X_WE_start, self.p_WC, num_pts=100, fig=None, ax=None
        )
        plt.show()


if __name__ == "__main__":
    unittest.main()
