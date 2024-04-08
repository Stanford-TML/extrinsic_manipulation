import open3d as o3d
import copy
from scipy.spatial.transform import Rotation as scipy_rot
import numpy as np


def normalize_two_pcds(pcd1, pcd2):
    """first move pcd1 and pcd2 so that pcd1 is centered at origin, then scale two point clouds to make them fit in (-1, 1)"""
    pcd1 = pcd1.copy()
    pcd2 = pcd2.copy()
    center = np.mean(pcd1, axis=0)
    pcd1 -= center
    pcd2 -= center
    scale = np.max(np.abs(np.concatenate([pcd1, pcd2], axis=0)))
    pcd1 /= scale
    pcd2 /= scale
    return pcd1, pcd2, center, scale



def transform_pc_points(pc_points, q_W, p_W, quat_convention):
    if quat_convention=='wxyz':
        # Drake's convention
        rot_mat_W = scipy_rot.from_quat(q_W[[1,2,3,0]]).as_matrix()
    elif quat_convention=='xywz':
        # Scipy's convensions
        rot_mat_W = scipy_rot.from_quat(q_W).as_matrix()
    pc_points = (rot_mat_W @ pc_points.T).T+p_W
    return pc_points

def convert_pc_to_mesh_alpha(pc, alpha, compute_convex_hull=True):
    """
    """
    raise DeprecationWarning
    alpha_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pc, alpha=alpha)
    if compute_convex_hull:
        return o3d.geometry.compute_mesh_convex_hull(alpha_mesh)
    return alpha_mesh

def convert_pc_to_mesh_poisson(pc, **kwargs):
    """
    """
    if not pc.has_normals():
        pc.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=6, **kwargs)[0]
    return poisson_mesh

def compute_pc_convex_mesh_and_maybe_save(pc, filename, mesh_scale=1., save=True):
    pc_scaled = copy.deepcopy(pc)
    pc_scaled.scale(mesh_scale, pc_scaled.get_center())
    mesh = pc_scaled.compute_convex_hull()[0]
    if save:
        o3d.io.write_triangle_mesh(filename, mesh)
    return mesh
