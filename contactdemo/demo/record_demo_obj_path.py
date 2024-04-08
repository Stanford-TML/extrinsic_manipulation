"""
This script uses megapose to collect a state trajectory of the object
The demonstration is a dictionary
{
    object: obj
    state_sequence: np.array([state1, state2, ...])
    contact_sequence: [(contact_type_1,state_idx_1), (contact_type_2,state_idx_2), ...]
}
For n contact types, there are n-1 contact switches
"""

import argparse
import numpy as np
from contactdemo.lib.drake.state_estimator import StateEstimator
from contactdemo.lib.utils.math_utils import X_to_pos_quat
from scipy.spatial.transform import Rotation as scipy_rot
import time
from isaacgymenvs.primitives.primitive_types import PrimitiveType, PrimitivePlan
import pygame
import pickle
import os


# hardcoded world-robot transform
X_WR = np.eye(4)
franka_start_pos = np.array(
    [
        -0.45,
        0.0,
        1.025,
    ]  # [-0.45, 0.0, 1.0 + table_thickness / 2 + table_stand_height]
)
franka_start_rot = scipy_rot.from_quat([0.0, 0.0, 0.0, 1.0]).as_matrix()
X_WR[:3, :3] = franka_start_rot
X_WR[:3, 3] = franka_start_pos


class SpoofStateEstimator:
    "for debugging purposes, returns the identity matrix as the object pose"

    def __init__(self):
        self.X_RB = np.eye(4)

    def get_object_X_RB(self, filter=False):
        return self.X_RB


def record_state_demo(
    sample_interval: float,
    object: str,
    first_primitive: PrimitiveType,
    debug=False,
):
    # adapted from https://stackoverflow.com/questions/50455952/processing-an-image-with-open-cv-every-n-seconds
    state_estimator = (
        StateEstimator(
            state_estimator_mode="megapose",
            ycb_idx=None,
            visualize=False,
            filter=False,
        )
        if not debug
        else SpoofStateEstimator()
    )
    result = dict()
    result["object"] = object
    state_seq = []
    current_idx = 0
    current_primitive = first_primitive
    contact_seq = []
    # For timing
    delta = 0
    previous = time.time()
    contact_seq.append((current_primitive, current_idx))
    while True:
        # Get the current time, increase delta and update the previous variable
        current = time.time()
        delta += current - previous
        previous = current

        # Check if 3 (or some other value) seconds passed
        if delta > sample_interval:
            # Reset the time counter
            delta = 0
            # get the state
            X_RB = state_estimator.get_object_X_RB(
                filter=filter,
            )
            X_WB = X_WR @ X_RB
            state_seq.append(X_WB)
            # get the contact type
            current_idx += 1
        # listen for key input
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                # Quit if 'q' is pressed
                if event.key == pygame.K_q:
                    result["state_sequence"] = np.array(state_seq)
                    result["contact_sequence"] = contact_seq
                    return result
                # Change the current primitive if a number is pressed
                try:
                    current_primitive = PrimitiveType(event.key - 48)
                    contact_seq.append((current_primitive, current_idx))
                    print("Current primitive is", current_primitive.name)
                except ValueError:
                    print("Invalid key")
        time.sleep(0.01)


if __name__ == "__main__":
    # Get command line arguments for object type and whether is mock
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--spoof_state",
        action="store_true",
    )
    parser.add_argument(
        "--save_primitive_plan",
        type=bool,
        default=True,
    )
    args = parser.parse_args()
    obj_str = args.object
    print(f"Recording a state trajectory for {obj_str}")
    print("Press a number to tag the start of each primitive")
    i = 0
    while 1:
        try:
            print(f"{i}: {PrimitiveType(i).name}")
        except ValueError:
            break
        i += 1
    print(
        "Set object at the initial state and press the first primitve number to start recording"
    )
    # Detect the first keypress with pygame
    pygame.init()
    # Hack for headless. The pygame window is not visible but it must be selected to detect keypresses
    screen = pygame.display.set_mode((1, 1))
    first_primitive = None
    while first_primitive is None:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                first_primitive = PrimitiveType(event.key - 48)
    print("Press 'q' to end recording\n")
    print("First primitive is", first_primitive.name)
    # wait for key input
    result = record_state_demo(0.2, obj_str, first_primitive, args.spoof_state)
    contact_seq = "_".join(
        [
            result["contact_sequence"][i][0].name
            for i in range(len(result["contact_sequence"]))
        ]
    )
    print("contact sequence", contact_seq)
    # make a directory called output
    os.makedirs("output", exist_ok=True)
    # pickle dump the result
    with open(f"outputs/{obj_str}_{contact_seq}_{time.time()}.pkl", "wb") as f:
        pickle.dump(result, f)

    # Convert and save as PrimitivePlan
    if args.save_primitive_plan:
        primitive_plan = []
        for i in range(len(result["contact_sequence"]) - 1):
            current_prim_type, start_idx = result["contact_sequence"][i]
            next_prim_type, end_idx = result["contact_sequence"][i + 1]
            prim = PrimitivePlan(
                start_pose_WB=np.concatenate(
                    X_to_pos_quat(result["state_sequence"][start_idx])
                ),
                goal_pose_WB=np.concatenate(
                    X_to_pos_quat(result["state_sequence"][end_idx])
                ),
                primitive_type=current_prim_type,
                max_num_steps=500,  # Arbitrary
                controller=None,
                control_type=None,
                demo_object=obj_str,
                next_primitive_type=next_prim_type,
            )
            primitive_plan.append(prim)
        # Save the last one
        current_prim_type, start_idx = result["contact_sequence"][-1]
        primitive_plan.append(
            PrimitivePlan(
                start_pose_WB=np.concatenate(
                    X_to_pos_quat(result["state_sequence"][start_idx])
                ),
                goal_pose_WB=np.concatenate(
                    X_to_pos_quat(result["state_sequence"][-1])
                ),
                primitive_type=current_prim_type,
                max_num_steps=500,  # Arbitrary
                controller=None,
                control_type=None,
                demo_object=obj_str,
            )
        )
        output_name = (
            f"outputs/{obj_str}_{contact_seq}_primitive_plan_{time.time()}.pkl"
        )
        with open(output_name, "wb") as f:
            pickle.dump(primitive_plan, f)
        print(f"saved to {output_name}")
