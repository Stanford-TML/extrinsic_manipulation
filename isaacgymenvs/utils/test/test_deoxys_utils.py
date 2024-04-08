import isaacgymenvs.utils.deoxys_utils as deoxys_utils
import unittest

import numpy as np
import time


class DeoxysRobotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.robot = deoxys_utils.DeoxysRobot(use_visualizer=True)

    def test_reset_joint_to_default(self):
        """
        Verify on hardware that the robot is in the correct pose
        """
        self.robot.set_q_to_default()
        # Get the robot pose
        np.testing.assert_allclose(
            self.robot.last_arm_q, self.robot.default_arm_q, atol=1e-3
        )

    def test_osc_move_displacement(self):
        self.robot.set_q_to_default()
        time.sleep(1)
        act = np.eye(7) * 0.1
        act[3:] *= 0.2
        act[:, -1] = 0.05
        for i in range(6):
            self.robot.osc_move_displacement(act[i], 2)
            time.sleep(1)
        self.robot.set_q_to_default()


if __name__ == "__main__":
    unittest.main()
