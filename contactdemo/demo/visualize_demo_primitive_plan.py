"""
Load the primitive plan and visualize in a Drake franak table station
"""

import argparse
import pickle
import os
import yaml
import glob
from contactdemo.lib.drake.franka_table_station import *
import time


def visualize_q(franka_station, q):
    diagram_context = franka_station.set_positions(q)
    franka_station.run_simulation(0.01, diagram_context)
    time.sleep(5)


def main(primitive_sequence_path):
    """
    Given an object pushing trajectory dataset with the object initial pose and
    object final pose, find the robot initial q with IK
    """
    # Load the dataset
    print("loading from ", primitive_sequence_path)
    with open(primitive_sequence_path, "rb") as pickle_file:
        primitive_plan = pickle.load(pickle_file)
    obj_str = primitive_plan[0].demo_object
    print("Object: ", obj_str)

    # Load the cfg for franka table station
    pwd = os.path.dirname(os.path.realpath(__file__))
    cfg_file = os.path.join(pwd, "../../../contactdemo/configs/franka_table_scene.yaml")
    with open(cfg_file, "r") as cfg_file:
        current_cfg = yaml.safe_load(cfg_file.read())

    obj_path = "/home/tml-franka-beast/exp/isaacgymenvs/assets/urdf/tml"
    # scan obj_path and find the matching object file
    folder_pattern = obj_path + "/" + obj_str + "*"
    obj_folder = glob.glob(folder_pattern)[0]  # There should only be one
    object_file = obj_folder + "/OBJ/3DModel.obj"
    franka_station = FrankaTableStation(current_cfg, object_file)
    for idx, prim in enumerate(primitive_plan):
        print("Primitive", idx, "is", prim.primitive_type.name)
        # Display the initial pose
        q0 = np.zeros(16)
        # Initialize the robot to the default arm q
        franka_station.get_quat_pos_WB(q0)[:] = (
            math_utils.scipy_pose_state_to_drake_pose_state(prim.start_pose_WB)
        )
        print("start pose")
        visualize_q(franka_station, q0)
    franka_station.get_quat_pos_WB(q0)[:] = (
        math_utils.scipy_pose_state_to_drake_pose_state(prim.goal_pose_WB)
    )
    print("goal pose")
    visualize_q(franka_station, q0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--primitive_sequence_path",
        help="Path to the state pair dataset",
        type=str,
        required=True,
    )
    # /home/tml-franka-beast/exp/contact_demo/contactdemo/data/1130_general_pushing.pkl
    args = parser.parse_args()
    main(args.primitive_sequence_path)
