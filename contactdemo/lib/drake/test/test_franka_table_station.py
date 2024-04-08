from contactdemo.lib.drake.franka_table_station import *
import yaml
import os
import time
import unittest

from pydrake.all import ExtractSimulatorConfig
from pydrake.geometry import AddContactMaterial, ProximityProperties
import pydrake.solvers as mp
from pydrake.math import RotationMatrix
from pydrake.common.eigen_geometry import Quaternion


def test_create_franka_table_plant():
    # Load the cfg
    pwd = os.path.dirname(os.path.realpath(__file__))

    cfg_file = os.path.join(pwd, "../../../configs/franka_table_scene.yaml")
    with open(cfg_file, "r") as cfg_file:
        current_cfg = yaml.safe_load(cfg_file.read())
    franka_station = FrankaTableStation(current_cfg)
    diagram_context = franka_station.set_positions(
        [-1.57, 0.1, 0, 0, 0, 1.6, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2]
    )
    # Note that drake pose convention is Quat[w,x,y,z] + Pos[x,y,z]

    franka_station.run_simulation(5.0, diagram_context)


class TestFrankaTableStationIK(unittest.TestCase):
    def setUp(self) -> None:
        pwd = os.path.dirname(os.path.realpath(__file__))
        cfg_file = os.path.join(pwd, "../../../configs/franka_table_scene.yaml")
        with open(cfg_file, "r") as cfg_file:
            self.cfg = yaml.safe_load(cfg_file.read())
        self.cfg["visualize"] = True
        object_file = "/home/tml-franka-beast/exp/isaacgymenvs/assets/urdf/ycb/003_cracker_box/poisson/textured.obj"
        self.franka_station = FrankaTableStation(self.cfg, object_file)
        return super().setUp()

    def _solve_ik_and_maybe_visualize(self, ik, q0=None):
        # Initial guess doesn't matter
        if q0 is None:
            q0 = np.array([-1.57, 0.1, 0, 0, 0, 1.6, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2])
        prog = ik.get_mutable_prog()
        prog.SetInitialGuess(ik.q(), q0)
        result = mp.Solve(prog)
        self.assertTrue(result.is_success())
        q_val = result.GetSolution(ik.q())
        diagram_context = self.franka_station.set_positions(q_val)
        if self.cfg["visualize"]:
            # Publish the
            self.franka_station.run_simulation(0.01, diagram_context)
            # Sleep so the simulation has time to load
            time.sleep(8)
        return result

    def test_add_object_env_distance_constraint_to_ik(self):
        ik = self.franka_station.construct_ik()
        self.franka_station.add_object_env_distance_constraint_to_ik(
            ik,
            0.00,
            0.001,
            "table",
        )
        # visually confirm object is in contact with the table
        self._solve_ik_and_maybe_visualize(ik)
        self.franka_station.add_object_env_distance_constraint_to_ik(
            ik,
            0.00,
            0.001,
            "wall",
        )
        # visually confirm object is in contact with the wall
        self._solve_ik_and_maybe_visualize(ik)

        # TODO: check the solution values

    def test_add_obj_point_to_env_point_distance_constraint_to_ik(self):
        ik = self.franka_station.construct_ik()
        # YCB cracker box has dimensinos 6*15.8*21cm
        # p_BC is on the center of the smallest face
        p_BC = np.array([0.0, 0.105, 0.0])  # This is chosen as the
        p_EC = np.array([0.0, 0.0, 0.0])
        self.franka_station.add_obj_point_to_env_point_distance_constraint_to_ik(
            ik, p_BC, p_EC, 0.00, 0.001, "table"
        )
        # visually confirm object is in contact with the table center
        self._solve_ik_and_maybe_visualize(ik)

        # TODO: check the solution values

    def test_add_obj_env_contact_constraints_to_ik(self):
        ik = self.franka_station.construct_ik()
        p_BC = np.array([0.0, 0.105, 0.0])  # This is chosen as the
        p_EC = np.array([0.0, 0.0, 0.0])
        # The box should be more or less upright on the table
        self.franka_station.add_obj_env_contact_constraints_to_ik(
            ik, p_BC, p_EC, 0.0, 0.001, 0.0, 0.001, "table"
        )
        self._solve_ik_and_maybe_visualize(ik)

    def test_add_corner_constraints_to_ik(self):
        ik = self.franka_station.construct_ik()
        p_BC = np.array([0.0, 0.105, 0.03])  # This is chosen as the
        p_EC = np.array([0.0, 0.0, 0.0])
        self.franka_station.add_object_env_distance_constraint_to_ik(
            ik,
            0.00,
            0.001,
            "table",
        )
        # visually confirm object is in contact with the table
        self.franka_station.add_object_env_distance_constraint_to_ik(
            ik,
            0.00,
            0.001,
            "wall",
        )
        self.franka_station.add_obj_point_to_env_point_distance_constraint_to_ik(
            ik, p_BC, p_EC, 0.00, 0.01, "wall"
        )
        # visually confirm object is in contact with both the wall and the table
        # plus the 2nd longest edge is in contact with the wall-table edge
        self._solve_ik_and_maybe_visualize(ik)

    def test_add_obj_R_WB_constraint(self):
        ik = self.franka_station.construct_ik()
        # No rotation
        self.franka_station.add_obj_R_WB_constraint(ik, RotationMatrix(), 0.01)
        self._solve_ik_and_maybe_visualize(ik)

        ik2 = self.franka_station.construct_ik()
        q = np.array([0.924, 0, 0.383, 0])
        # 45 degree rotation around y axis
        q /= np.linalg.norm(q)
        self.franka_station.add_obj_R_WB_constraint(
            ik2, RotationMatrix(Quaternion(*q)), 0.05
        )
        self._solve_ik_and_maybe_visualize(ik2)

    def test_add_fingertip_to_object_distance_constraint_to_ik(self):
        ik = self.franka_station.construct_ik()
        self.franka_station.add_object_env_distance_constraint_to_ik(
            ik,
            0.00,
            0.001,
            "table",
        )
        self.franka_station.add_object_env_distance_constraint_to_ik(
            ik,
            0.00,
            0.001,
            "wall",
        )

        self.franka_station.add_fingertip_to_object_distance_constraint_to_ik(
            ik, RobotFinger.LEFT, 0.00, 0.001
        )
        self._solve_ik_and_maybe_visualize(ik)
        self.franka_station.add_fingertip_to_object_distance_constraint_to_ik(
            ik, RobotFinger.RIGHT, 0.00, 0.001
        )
        self._solve_ik_and_maybe_visualize(ik)

    def test_add_obj_p_WB_quadratic_cost(self):
        ik = self.franka_station.construct_ik()
        self.franka_station.add_obj_p_WB_quadratic_cost(ik, np.array([0.0, 1.0, 1.0]))
        # Visually verify the position is as expected
        self._solve_ik_and_maybe_visualize(ik)

    def test_add_obj_R_WB_cost(self):
        q = np.array([0.924, 0, 0.383, 0])
        # 45 degree rotation around y axis
        q /= np.linalg.norm(q)
        for c in [0.0, 1.0]:
            # Try with and without cost
            # Object should rotate only with cost
            ik = self.franka_station.construct_ik()
            self.franka_station.add_obj_R_WB_cost(ik, RotationMatrix(Quaternion(*q)), c)
            self._solve_ik_and_maybe_visualize(ik)

    def test_pushing_ik(self):
        ik = self.franka_station.construct_ik()
        p_BC = np.array([0.0, 0.105, 0.03])  # This is chosen as the
        p_EC = np.array([0.0, 0.0, 0.0])
        self.franka_station.add_object_env_distance_constraint_to_ik(
            ik,
            0.00,
            0.001,
            "table",
        )
        # visually confirm object is in contact with the table
        self.franka_station.add_object_env_distance_constraint_to_ik(
            ik,
            0.00,
            0.001,
            "wall",
        )
        # visually confirm object is in contact with both the wall and the table
        # plus the 2nd longest edge is in contact with the wall-table edge
        self._solve_ik_and_maybe_visualize(ik)

    def test_solve_object_pose_scene_with_contact(self):
        p_BC_list = [
            np.array([-0.028, 0.0, 0.0]),  # center of largest face on cheezeit box
            np.array([0.0, 0.0, 0.1]),  # top of cheezit box
            np.array([0.0, 0.0, -0.1]),  # bottom of cheezit box
        ]
        p_EC_list = [
            # TODO
            np.array([0.0, 0.0, 0.0]),  # center of the table
            np.array([0.0, 0.0, 0.0]),  # center of the table
            np.array([0.0, 0.0, 0.0]),  # center of the table
        ]
        contact_object_list = [
            FrankaTableSceneObject.TABLE,
            FrankaTableSceneObject.WALL,
            FrankaTableSceneObject.ROBOT,
        ]
        self.franka_station.solve_object_pose_scene_with_contact(
            p_BC_list,
            p_EC_list,
            contact_object_list,
            visualize=True,
            return_q=True,
        )


if __name__ == "__main__":
    unittest.main()
    # test_create_franka_table_plant()
