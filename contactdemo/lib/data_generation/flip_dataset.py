import pickle
import transforms3d
from tqdm import tqdm
import numpy as np
from contactdemo.lib.data_generation.utils.pybullet_env import init_env
import pybullet as p
from contactdemo.lib.data_generation.utils.scanned_utils import create_scanned_asset


if __name__ == "__main__":
    original_dataset_path = "contactdemo/data/pushing/0304_merged.pkl"
    with open(original_dataset_path, "rb") as fin:
        dataset = pickle.load(fin)

    new_dataset = {}
    obj_names = [
        "cereal",
        "cocoa",
        "cracker",
        "flapjack",
        "oat",
        "seasoning",
        "wafer",
        "camera",
        "meat",
        "onion",
    ]
    obj_names2axis = {
        # 'cereal': [0, 1],
        # 'cocoa': [1, 2],
        # 'cracker': [0, 1],
        # 'flapjack': [0, 1],
        # 'oat': [0, 1],
        # 'seasoning': [0, 1],
        # 'wafer': [0, 1],
        # 'cameratape': [1, 2],
        # 'meat': [0, 1],
        "oniontape": [0, 1],
    }
    # _ = p.connect(p.GUI)
    resimulate = True
    if resimulate:
        _ = p.connect(p.DIRECT)
        env_info = init_env()

    asset_id = -1
    ct = 0
    reset_every_n_examples = 1000
    for obj_name, axes in obj_names2axis.items():
        new_dataset[obj_name] = []
        print(obj_name)
        for data in tqdm(dataset[obj_name][:10]):
            # new_dataset[obj_name].append(data)
            # flip the datapoint
            new_data = data.copy()
            initial_pos, initial_quat = (
                new_data["initial_state"][:3],
                new_data["initial_state"][3:],
            )
            final_pos, final_quat = (
                new_data["final_state"][:3],
                new_data["final_state"][3:],
            )

            if resimulate:
                if ct % reset_every_n_examples == 0:
                    p.resetSimulation()
                    env_info = init_env()
                    wall_bbox = env_info["wall_bbox"]
                    table_bbox = env_info["table_bbox"]
                    asset_id = -1
                if asset_id != -1:
                    p.removeBody(asset_id)
                asset_id = create_scanned_asset(
                    obj_name,
                    use_fix_base=False,
                )

            ct += 1

            for axis in axes:

                # flip the object
                vec = np.zeros(3)
                vec[axes[0]] = 1
                flip_quat = transforms3d.quaternions.axangle2quat(vec, np.pi)

                initial_quat = transforms3d.quaternions.qmult(
                    np.array(initial_quat)[[3, 0, 1, 2]], flip_quat
                )[[1, 2, 3, 0]]
                final_quat = transforms3d.quaternions.qmult(
                    np.array(final_quat)[[3, 0, 1, 2]], flip_quat
                )[[1, 2, 3, 0]]

                if resimulate:
                    # resimulate the object
                    p.setGravity(0, 0, -9.8)
                    p.resetBasePositionAndOrientation(
                        asset_id, np.array(initial_pos), np.array(initial_quat)
                    )

                    def get_equilibrium(asset_id):
                        max_equilibrium_steps = 500
                        for _ in range(max_equilibrium_steps):
                            p.stepSimulation()
                            vel, _ = p.getBaseVelocity(asset_id)
                            if np.linalg.norm(vel) < 0.01:
                                return True
                        return False

                    drop_is_eq = get_equilibrium(asset_id)
                    if not drop_is_eq:
                        continue

                    initial_pos, initial_quat = p.getBasePositionAndOrientation(
                        asset_id
                    )
                    p.resetBasePositionAndOrientation(
                        asset_id, np.array(final_pos), np.array(final_quat)
                    )
                    drop_is_eq = get_equilibrium(asset_id)
                    if not drop_is_eq:
                        continue

                    final_pos, final_quat = p.getBasePositionAndOrientation(asset_id)
                new_dataset[obj_name].append(
                    {
                        "initial_state": list(initial_pos) + list(initial_quat),
                        "final_state": list(final_pos) + list(final_quat),
                        "obj_name": obj_name,
                    }
                )

    new_dataset_path = "contactdemo/data/pushing/0304_merged_flipped.pkl"
    with open(new_dataset_path, "wb") as fout:
        pickle.dump(new_dataset, fout)

    print("dataset saved to", new_dataset_path)
