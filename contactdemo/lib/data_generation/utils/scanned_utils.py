import tempfile
import os
import glob
from contactdemo.lib.data_generation.utils.fps import furthest_point_sample
from contactdemo.lib.data_generation.utils.coacd_utils import convex_decompose_coacd
import numpy as np
import trimesh
import open3d as o3d

current_dir_path = os.path.dirname(os.path.realpath(__file__))

scanned_root = os.path.join(current_dir_path, "../../../assets/object_scans")


def load_obj(
    visual_geometry_path,
    collision_geometry_path,
    scale,
    tmp_dir="/tmp",
    start_pos=(0, 0, 0),
    use_fix_base=True,
    geometry_origin=None,
):
    import pybullet as p

    # create temporary urdf file
    tmp = tempfile.NamedTemporaryFile(
        mode="w", dir=tmp_dir, suffix=".urdf", delete=False
    )
    #     tmp.writelines(
    #         """
    # <robot name="simple_box_robot">
    #     <link name="box_link">
    #         <visual>
    #             <geometry>
    #                 <box size="0.1 0.1 0.2"/>
    #             </geometry>
    #             <material name="blue">
    #                 <color rgba="0 0 1 1"/>
    #             </material>
    #         </visual>
    #         <collision>
    #             <geometry>
    #                 <box size="0.1 0.1 0.2"/>
    #             </geometry>
    #         </collision>
    #     </link>
    # </robot>
    # """.split(
    #             "\n"
    #         )
    #     )
    if geometry_origin is None:
        geometry_origin = [0, 0, 0]
    tmp.writelines(
        [
            '<?xml version="1.0" ?>',
            '<robot name="object" xmlns:xacro="http://www.ros.org/wiki/xacro">',
            '  <link name="rigid_body">',
            "<inertial>",
            f' <origin xyz="{" ".join(map(str, geometry_origin))}" rpy="0 0 0"/>',
            '  <mass value="0.1"/>',
            ' <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>',
            "</inertial>",
            "    <visual>",
            "      <geometry>",
            '        <mesh filename="package://{path}" scale="{scale} {scale} {scale}"/>'.format(
                path=visual_geometry_path, scale=scale
            ),
            "      </geometry>",
            "    </visual>",
            "    <collision>",
            "      <geometry>",
            '        <mesh filename="package://{path}" scale="{scale} {scale} {scale}"/>'.format(
                path=collision_geometry_path, scale=scale
            ),
            "      </geometry>",
            "    </collision>",
            "  </link>",
            "</robot>",
        ]
    )
    tmp.close()

    return p.loadURDF(tmp.name, start_pos, useFixedBase=use_fix_base)


def get_ycb_pcd(ycb_idx, model="poisson", n_samps=1024):
    ycb_idx_str = str(ycb_idx).zfill(3)
    ycb_root = "ycb"
    folder_pattern = ycb_root + "/" + ycb_idx_str + "*"
    ycb_folder = glob.glob(folder_pattern)[0]  # There should only be one
    cache_path = os.path.join("ycb_pcd_fps_cache", f"{ycb_idx}_{model}_{n_samps}.npz")
    if os.path.exists(cache_path):
        return np.load(cache_path)["pcd"]
    else:
        import open3d as o3d

        point_cloud = o3d.io.read_point_cloud(
            ycb_folder + f"/{model}/" + "nontextured.ply"
        )
        pcd = furthest_point_sample(np.asarray(point_cloud.points), n_samps)
        np.savez(cache_path, pcd=pcd)
        return pcd


geometry_center_dict = {}


def create_scanned_asset(
    obj_name,
    start_pos=[0, 0, 0],
    use_fix_base=True,
):
    global geometry_center_dict
    global scanned_root
    global ycb_root_albert
    folder_pattern = scanned_root + "/" + obj_name
    scanned_folder = glob.glob(folder_pattern)[0]  # There should only be one
    key = obj_name
    obj_path = scanned_folder + f"/OBJ/3DModel.obj"
    if key not in geometry_center_dict:
        mesh = trimesh.load(obj_path)
        geometry_center_dict[key] = np.mean(mesh.vertices, axis=0)

    coacd_obj_path = os.path.join(scanned_folder, "OBJ",  "textured_coacd.obj")
    if not os.path.exists(coacd_obj_path):
        convex_decompose_coacd(
            os.path.join(obj_path),
            os.path.join(os.path.join(scanned_folder, "OBJ",  "textured_coacd.obj")),
        )

    visual_geometry_path = obj_path
    collision_geometry_path = coacd_obj_path
    asset_id = load_obj(
        visual_geometry_path,
        collision_geometry_path,
        1.0,
        start_pos=start_pos,
        use_fix_base=use_fix_base,
        geometry_origin=geometry_center_dict[key],
    )
    return asset_id


mesh_dict = {}


def sample_pcd(
    obj_name, n_samps=1024
):
    raise NotImplementedError
    global mesh_dict
    if ycb_idx not in mesh_dict:
        global ycb_geometry_center_dict
        global ycb_root
        global ycb_root_albert
        ycb_idx_str = str(ycb_idx).zfill(3)
        if use_albert_scanned:
            folder_pattern = ycb_root_albert + "/" + ycb_idx_str + "*"
        else:
            folder_pattern = ycb_root + "/" + ycb_idx_str + "*"
        ycb_folder = glob.glob(folder_pattern)[0]  # There should only be one

        visual_geometry_path = os.path.join(ycb_folder, mesh_type + "/textured.obj")
        mesh_dict[ycb_idx] = o3d.io.read_triangle_mesh(visual_geometry_path)
    mesh = mesh_dict[ycb_idx]
    pcd = np.array(o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, n_samps).points)
    pcd = pcd - ycb_geometry_center_dict[(ycb_idx, mesh_type)]
    return pcd
