from contactdemo.lib.data_generation.utils.pybullet_env import init_env
from contactdemo.lib.data_generation.utils.scanned_utils import create_scanned_asset

import os
import json
import random
import pybullet as p
import time
import numpy as np
import pickle
import transforms3d
import argparse


def visualize_init_final(pkl_path, verbose=False):
    _ = p.connect(p.GUI)
    env_info = init_env()

    asset_idx = -1
    with open(pkl_path, "rb") as fin:
        dataset_all = pickle.load(fin)

    dataset = []
    for k, v in dataset_all.items():
        random.shuffle(v)
        dataset += v[:2]

    for data in dataset:
        if asset_idx != -1:
            # remove the object from the environment
            p.removeBody(asset_idx)

        asset_idx = create_scanned_asset(
            data["obj_name"],
            use_fix_base=True,
        )
        p.setGravity(0, 0, -9.8)
        p.resetBasePositionAndOrientation(
            asset_idx,
            np.array(data["initial_state"][:3]),
            data["initial_state"][3:7],
        )

        for i in range(20):
            p.stepSimulation()
            time.sleep(1 / 240)

        p.resetBasePositionAndOrientation(
            asset_idx, data["final_state"][:3], data["final_state"][3:7]
        )

        for i in range(20):
            p.stepSimulation()
            time.sleep(1 / 240)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", type=str, default="contactdemo/data/pushing/demo.pkl")
    args = parser.parse_args()
    visualize_init_final(args.pkl_path, verbose=False)
