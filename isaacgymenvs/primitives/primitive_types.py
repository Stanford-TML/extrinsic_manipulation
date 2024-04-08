import enum

from dataclasses import dataclass
import typing

from abc import ABC
import numpy as np


class PrimitiveType(enum.Enum):
    Push = 0
    Pivot = 1
    Relocate = 2
    Grasp = 3
    Withdraw = 4  # Bookkeeping only, this is basically relocate
    Pull = 5


@dataclass
class PrimitivePlan:
    start_pose_WB: typing.Union[np.ndarray, None, list]
    goal_pose_WB: typing.Union[np.ndarray, None, list]
    primitive_type: PrimitiveType
    max_num_steps: typing.Union[int, None]
    controller: typing.Union[ABC, None]
    control_type: str
    start_robot_q: typing.Union[np.ndarray, None, list] = None
    goal_robot_q: typing.Union[np.ndarray, None, list] = None
    next_primitive_type: typing.Union[PrimitiveType, None] = (
        None  # This excludes "Relocate," which does not require interacting with the object
    )
    demo_object: typing.Union[str, None] = None
