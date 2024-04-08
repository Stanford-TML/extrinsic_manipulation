import argparse
import json
import os
import random
import time

import numpy as np
import pybullet as p
import transforms3d
from scipy.linalg import qr
from tqdm import tqdm

from contactdemo.lib.data_generation.utils.pybullet_env import init_env
from contactdemo.lib.data_generation.utils.scanned_utils import create_scanned_asset


def random_sample_rotation_matrix():
    # Reference https://math.stackexchange.com/questions/442418/random-generation-of-rotation-matrices
    # Generate a 3x3 matrix with normally distributed random numbers
    random_matrix = np.random.randn(3, 3)

    # Perform QR decomposition
    Q, R = qr(random_matrix)

    # Adjust the sign of the first column
    Q[:, 0] = Q[:, 0] * (2 * (np.random.rand() > 0.5) - 1)

    # Adjust the second column to ensure a positive determinant
    if np.linalg.det(Q) < 0:
        Q[:, 1] = -Q[:, 1]

    return Q


def main(dataset_path, n_examples, start_idx, seed, verbose=False):
    if verbose:
        _ = p.connect(p.GUI)
    else:
        _ = p.connect(p.DIRECT)

    os.makedirs(dataset_path, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)

    asset_idx = -1
    ct = 0

    progrss_bar = tqdm(range(n_examples))
    reset_every_n_examples = 1000

    scan_root = 'contactdemo/assets/object_scans'
    obj_names = [f for f in os.listdir(scan_root) if os.path.isdir(os.path.join(scan_root, f))]
    # oat, cracker, wafer, cereal, cocoa
    # obj_names = ['oat', 'cracker', 'wafer', 'cereal']
    # obj_names = ['cocoa']
    # obj_names = ['onion', 'flapjack']
    # obj_names = ['cameratape', 'oniontape']

    n_examples_each_obj = int(np.ceil(n_examples / len(obj_names)))
    target_obj_names = []
    for obj_name in obj_names:
        target_obj_names += [obj_name] * n_examples_each_obj
    random.shuffle(target_obj_names)

    exs = []

    while True:
        if ct % reset_every_n_examples == 0:
            p.resetSimulation()
            env_info = init_env()
            wall_bbox = env_info["wall_bbox"]
            table_bbox = env_info["table_bbox"]
        if asset_idx != -1:
            # remove the object from the environment
            p.removeBody(asset_idx)

        obj_name = target_obj_names[ct]
        asset_idx = create_scanned_asset(
            obj_name,
            use_fix_base=False,
        )

        # enable gravity
        p.setGravity(0, 0, -9.8)

        def random_put_object_on_table(asset_idx, reference_q=None):
            p.resetBasePositionAndOrientation(asset_idx, [0, 0, 0], [0, 0, 0, 1])
            aabb_min, aabb_max = p.getAABB(asset_idx)
            corners = [
                aabb_min,
                [aabb_min[0], aabb_min[1], aabb_max[2]],
                [aabb_min[0], aabb_max[1], aabb_min[2]],
                [aabb_min[0], aabb_max[1], aabb_max[2]],
                [aabb_max[0], aabb_min[1], aabb_min[2]],
                [aabb_max[0], aabb_min[1], aabb_max[2]],
                [aabb_max[0], aabb_max[1], aabb_min[2]],
                aabb_max,
            ]

            # sample a random rotation
            if reference_q is None:
                q_mat = random_sample_rotation_matrix()
            else:
                q_mat = transforms3d.quaternions.quat2mat(np.array(reference_q)[[3, 0, 1, 2]])
                jitter_range = np.pi / 6
                random_jitter = (np.random.random() * 0.5 - 1) * jitter_range
                axis = [0, 0, 1]
                q_mat = transforms3d.axangles.axangle2mat(axis, random_jitter) @ q_mat

            # get the aabb of the object after rotation
            corners_q = np.matmul(q_mat, np.array(corners).T).T
            aabb_min_q = np.min(corners_q, axis=0)

            q = transforms3d.quaternions.mat2quat(q_mat)[[1, 2, 3, 0]]

            # q = transforms3d.quaternions.axangle2quat([0, 1, 0], np.pi / 2)
            # sample a random position on the table
            margin = [0.1, 0.1, 0.001]
            max_iter = 100
            iters = 0
            old_pos, _ = p.getBasePositionAndOrientation(asset_idx)
            while iters < max_iter:
                pos = np.array(
                    [
                        random.uniform(
                            table_bbox[0] + margin[0], table_bbox[1] - margin[0]
                        ),
                        random.uniform(
                            table_bbox[2] + margin[0], table_bbox[3] - margin[1]
                        ),
                        table_bbox[5] + margin[2] - aabb_min_q[2],
                    ]
                )
                iters += 1
                if np.linalg.norm(pos - old_pos) <= 0.1 and np.linalg.norm(pos - old_pos) >= 0.02:
                    # if np.linalg.norm(pos - old_pos) >= 0.1 and np.linalg.norm(pos - old_pos) <= 0.2:
                    break

            p.resetBasePositionAndOrientation(asset_idx, pos, q)

            # run the simulation for 100 steps
            # while True:
            max_equilibrium_steps = 500

            if verbose:
                print("Dropping the object on the table")

            drop_is_eq = False
            for _ in range(max_equilibrium_steps):
                p.stepSimulation()

                vel, _ = p.getBaseVelocity(asset_idx)
                if np.linalg.norm(vel) < 0.01:
                    drop_is_eq = True
                    break

            return drop_is_eq

        drop_is_eq = random_put_object_on_table(asset_idx)
        if not drop_is_eq:
            continue

        if verbose:
            print("Initial position")

            for i in range(20):
                p.stepSimulation()
                time.sleep(1.0 / 240.0)

        pushing_initial_pos, pushing_initial_q = p.getBasePositionAndOrientation(
            asset_idx
        )

        # drop the object on the table again
        drop_is_eq = random_put_object_on_table(asset_idx, reference_q=pushing_initial_q)
        if not drop_is_eq:
            continue

        if verbose:
            print("Final position")
            for i in range(20):
                p.stepSimulation()
                time.sleep(1.0 / 240.0)

        pushing_final_pos, pushing_final_q = p.getBasePositionAndOrientation(asset_idx)

        # for pushing the CoM's height should not change
        invalid_push = abs(pushing_initial_pos[2] - pushing_final_pos[2]) > 0.01

        # we also filter out the case where the CoM is too high, which makes the pushing unstable
        obj_name2maximum_height = {'biscuit': 1.07, 'cereal': 1.07, 'chocolate': 1.055, 'cocoa': 1.07, 'cracker': 1.07,
                                   'gelatin': 1.05, 'meat': 1.06, 'mustard': 1.06, 'oat': 1.07, 'ramen': 1.07,
                                   'seasoning': 1.06, 'wafer': 1.06, 'flapjack': 1.06, 'camera': 1.052, 'oniontape': 1.065}
        if obj_name in obj_name2maximum_height:
            invalid_push = invalid_push or (
                    pushing_final_pos[2] > obj_name2maximum_height[obj_name]
            ) or pushing_initial_pos[2] > obj_name2maximum_height[obj_name]

        if obj_name == 'onion':
            invalid_push = invalid_push or (
                    pushing_final_pos[2] <= 1.085 and pushing_final_pos[2] >= 1.07)

        # get the closest points between the object and the wall
        contacts = p.getClosestPoints(asset_idx, env_info["wall_id"], 0.01)
        # we filter out the case where the object is in contact with the wall
        invalid_push = invalid_push or len(contacts) > 0

        if invalid_push:
            #print("Invalid push, skipping", obj_name)
            continue

        d = {
            "obj_name": obj_name,
        }
        d.update(
            {
                "initial_state": list(pushing_initial_pos)
                                 + list(pushing_initial_q),
                "final_state": list(pushing_final_pos) + list(pushing_final_q),
            }
        )
        exs.append(d)

        ct += 1
        progrss_bar.update(1)
        if ct >= n_examples:
            break

    with open(os.path.join(dataset_path, f"{start_idx}_{start_idx + n_examples}.json"), "w") as f:
        json.dump(exs, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--n_examples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--start_idx", type=int, default=0)

    args = parser.parse_args()
    main(args.dataset_path, args.n_examples, args.start_idx, args.seed, args.verbose)
