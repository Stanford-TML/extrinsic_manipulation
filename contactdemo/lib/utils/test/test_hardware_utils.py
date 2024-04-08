from contactdemo.lib.utils.hardware_utils import update_cfg_wall_pos_orn

from contactdemo.lib.drake.franka_table_station import FrankaTableStation

import time
import unittest
import os
import yaml
import numpy as np

from pydrake.math import RotationMatrix
from pydrake.common.eigen_geometry import Quaternion


class TestFrankaTableStationIK(unittest.TestCase):
    def setUp(self) -> None:
        pwd = os.path.dirname(os.path.realpath(__file__))
        self.cfg_file = os.path.join(pwd, "../../../configs/franka_table_scene.yaml")

    def test_update_cfg_wall_pos_orn(self):
        for left_x in [0.7, 0.75, 0.8]:
            for right_x in [0.7, 0.75, 0.8]:
                with open(self.cfg_file, "r") as cfg_file:
                    cfg = yaml.safe_load(cfg_file.read())
                cfg["visualize"] = True
                dut = update_cfg_wall_pos_orn(left_x, right_x, cfg)
                franka_station = FrankaTableStation(dut)
                franka_station.visualize_q(np.zeros(9))
                print("Wall pos", cfg["wall"]["wall_pos"])
                print("Wall rpy", cfg["wall"]["wall_rpy"])
                # Validate visually
                time.sleep(10)


if __name__ == "__main__":
    unittest.main()
    # test_create_franka_table_plant()
