from contactdemo.lib.utils.o3d_utils import *
import contactdemo.lib.utils.math_utils as math_utils
from rpl_vision_utils.configs.hardware_config import *
from rpl_vision_utils.networking.tracker_redis_interface import TrackerRedisSubInterface
from rpl_vision_utils.networking.pose_redis_interface import PoseRedisInterface
from extrinsic_manip.utils.ycb_utils import *
import contactdemo.lib.drake.franka_table_station as franka_table_station
import contactdemo.lib.drake.contact_retargeting as contact_retargeting

import numpy as np
import time


class StateEstimator:
    def __init__(
        self, state_estimator_mode="icp", object_type="ycb", visualize=True, **kwargs
    ) -> None:
        self.state_estimator_mode = state_estimator_mode
        self.franka_station = None
        self.refine_pose_with_ik = False
        self.X_WR = None
        self.X_RW = None
        assert state_estimator_mode == "megapose"
        self.pose_sub_interface = PoseRedisInterface(
            redis_host=REDIS_HOST, redis_port=REDIS_PORT
        )
        self.tracker_sub_interface = None
        self.last_X_RB = np.eye(4)
        if "drake_scene_cfg" in kwargs and "object_file_path" in kwargs:
            self.franka_station = franka_table_station.FrankaTableStation(
                kwargs["drake_scene_cfg"], kwargs["object_file_path"]
            )
            self.refine_pose_with_ik = True
            self.X_WR = self.franka_station.X_WR
            self.X_RW = np.linalg.inv(self.X_WR)

    def __del__(self) -> None:
        if self.tracker_sub_interface:
            self.tracker_sub_interface.close()

    def _get_object_pcd(self):
        """
        Get the point cloud of the object
        """
        pcd = self.tracker_sub_interface.get_result(self.instance_id)
        return pcd

    def update_vis(self, X_RB, X_RB_filtered, object_pcd_o3d):
        self.visualizer.update_obj(
            0,
            object_pcd_o3d,
        )
        self.visualizer.update_obj(
            1,
            copy.deepcopy(self.object_model_pcd_o3d).transform(
                X_RB_filtered,
            ),
        )

    def get_object_X_RB(
        self,
        R_RB_guess=None,
        p_RB_guess=None,
        run_global_registration=True,
        return_pcd=False,
        filter=True,
        block_on_timeout=True,
        return_last_pose=False,
        stale_state_timeout=1.0,
    ):
        """
        Get the object pose in the robot base frame. The frame is defined as follows:
        +X: forward direction of the robot (towards the table)
        +Y: starboard side of the robot
        +Z: table surface pointing up

        Get a camera observation of the object, then run ICP to get the object state
        :param X_RB_guess: Previous object pose in world frame
        """
        if self.state_estimator_mode == "megapose":
            while 1:
                object_pose_dict = self.pose_sub_interface.get_pose()
                if (
                    time.time() - object_pose_dict.get("timestamp")
                    > stale_state_timeout
                ):
                    print("Stale state")
                    if block_on_timeout:
                        time.sleep(0.5)
                        continue
                    else:
                        if return_last_pose:
                            return self.last_X_RB
                        return None
                object_pose = np.array(object_pose_dict.get("pose")).squeeze()
                self.last_X_RB = math_utils.pos_quat_to_X(
                    object_pose[:3], object_pose[3:]
                )
                if self.refine_pose_with_ik:
                    self.last_X_RB = (
                        self.X_RW
                        @ contact_retargeting.compute_object_X_WB_on_table(
                            self.franka_station, self.X_WR @ self.last_X_RB
                        )
                    )
                return self.last_X_RB
        else:
            raise NotImplementedError
