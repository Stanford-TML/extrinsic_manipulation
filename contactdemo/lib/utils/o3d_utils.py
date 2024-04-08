import open3d as o3d
import numpy as np
import copy


class PointCloudPersistentVisualizer:
    def __init__(self, pc_list, color_list=None) -> None:
        self.pc_list = []
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.color_list = color_list
        if self.color_list is None:
            self.color_list = [[0, 0, 1]] * len(pc_list)
        for idx, pc in enumerate(pc_list):
            self.add_pc(pc, self.color_list[idx])
        self.refresh()

    def __del__(self):
        self.vis.destroy_window()

    def add_pc(self, pc, color=[0, 0, 1]):
        self.pc_list.append(copy.deepcopy(pc))
        self.pc_list[-1].paint_uniform_color(color)
        self.vis.add_geometry(self.pc_list[-1])
        self.refresh()

    def clear(self):
        for pc in self.pc_list:
            self.vis.remove_geometry(pc)
        self.pc_list.clear()
        self.color_list.clear()
        self.refresh()

    def update_obj(self, pc_idx, new_pc):
        self.pc_list[pc_idx].points = new_pc.points
        self.vis.update_geometry(self.pc_list[pc_idx])
        self.refresh()

    def refresh(self):
        self.vis.poll_events()
        self.vis.update_renderer()


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries(
        [source_temp, target_temp],
        # zoom=0.4459,
        # front=[0.9288, -0.2951, -0.2242],
        # lookat=[1.6784, 2.0612, 1.4451],
        # up=[-0.3402, -0.9189, -0.1996],
    )


def make_o3d_pcd(point_cloud_np, labels_np=None, offset=None, label_colors=None):
    # Create Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    if offset is not None:
        point_cloud_np += offset
    point_cloud.points = o3d.utility.Vector3dVector(point_cloud_np)

    # Assign colors to points based on labels
    colors = np.zeros_like(point_cloud_np)
    if label_colors is None:
        label_colors = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    colors[:] = label_colors[0]
    if labels_np is not None:
        colors[labels_np == 2] = label_colors[2]
        colors[labels_np == 1] = label_colors[1]
        colors[labels_np == 0] = label_colors[0]
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud


def preprocess_point_cloud(pcd, voxel_size):
    # Adapted from http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html#RANSAC
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )

    return pcd_down, pcd_fpfh


def execute_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, voxel_size, method="ransac"
):
    # Adapted from http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html#RANSAC
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    if method == "ransac":
        result = (
            o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_down,
                target_down,
                source_fpfh,
                target_fpfh,
                True,
                distance_threshold,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                3,
                [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                        0.9
                    ),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                        distance_threshold
                    ),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(
                        np.pi / 2
                    ),
                ],
                o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
            )
        )
    elif method == "fgr":
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source_down,
            target_down,
            source_fpfh,
            target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold
            ),
        )
    else:
        raise NotImplementedError
    return result


def filter_object_pcd_and_make_o3d(object_pcd):
    num_pt_in_pc = len(object_pcd)
    object_pcd_o3d = make_o3d_pcd(object_pcd)
    # Remove outliers
    object_pcd_o3d, keep_idxs = object_pcd_o3d.remove_radius_outlier(
        nb_points=16, radius=0.03
    )
    object_pcd_o3d, keep_idxs = object_pcd_o3d.remove_statistical_outlier(
        nb_neighbors=int(0.05 * num_pt_in_pc), std_ratio=1.5
    )
    # print(f"Removed {num_pt_in_pc-len(keep_idxs)} outliers")
    # if len(keep_idxs) > 0:
    #     print("Range of filtered cloud", np.ptp(object_pcd_o3d.points, axis=0))
    return object_pcd_o3d
