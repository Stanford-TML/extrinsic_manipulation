import isaacgym.gymapi as gymapi
import tempfile
import os
import glob
import enum

current_dir_path = os.path.dirname(os.path.realpath(__file__))

ycb_root = os.path.join(current_dir_path, "../../assets/urdf/ycb")
tml_root = os.path.join(current_dir_path, "../../assets/urdf/tml")


def load_obj(gym, sim, asset_root, asset_file, scale, asset_options, tmp_dir="/tmp"):
    # create temporary urdf file
    asset_path = os.path.join(asset_root, asset_file)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", dir=tmp_dir, suffix=".urdf", delete=False
    )
    tmp.writelines(
        [
            '<?xml version="1.0" ?>',
            '<robot name="object" xmlns:xacro="http://www.ros.org/wiki/xacro">',
            '  <link name="rigid_body">',
            "    <visual>",
            "      <geometry>",
            '        <mesh filename="package://{path}" scale="{scale} {scale} {scale}"/>'.format(
                path=asset_path, scale=scale
            ),
            "      </geometry>",
            "    </visual>",
            "    <collision>",
            "      <geometry>",
            '        <mesh filename="package://{path}" scale="{scale} {scale} {scale}"/>'.format(
                path=asset_path, scale=scale
            ),
            "      </geometry>",
            "    </collision>",
            "  </link>",
            "</robot>",
        ]
    )
    tmp.close()

    return gym.load_asset(sim, "", tmp.name, asset_options)


def get_ycb_asset_path(ycb_idx, mesh_type=None):
    raise DeprecationWarning
    ycb_idx_str = str(ycb_idx).zfill(3)
    folder_pattern = ycb_root + "/" + ycb_idx_str + "*"
    ycb_folder = glob.glob(folder_pattern)[0]  # There should only be one
    if mesh_type is None:
        return ycb_folder
    return ycb_folder + "/" + mesh_type + "/textured.obj"


def create_ycb_asset(gym, sim, ycb_idx, mesh_type="google_16k", asset_options=None):
    raise DeprecationWarning
    ycb_folder = get_ycb_asset_path(ycb_idx)
    if asset_options is None:
        asset_options = gymapi.AssetOptions()
        asset_options.density = 1e3
        asset_options.override_inertia = True
        asset_options.angular_damping = 0.1
        asset_options.linear_damping = 0.1
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.resolution = 300000
        asset_options.vhacd_params.max_convex_hulls = 10
        asset_options.vhacd_params.max_num_vertices_per_ch = 64
    asset = load_obj(
        gym, sim, ycb_folder, mesh_type + "/textured.obj", 1.0, asset_options
    )
    return asset


class TMLObjectType(enum.Enum):
    """
    Wafer
    Chocolate
    Meat
    Seasoning
    Ramen
    Mustard
    Milk
    Gelatin
    Cracker
    """

    Wafer = "wafer"
    Chocolate = "chocolate"
    Meat = "meat"
    Seasoning = "seasoning"
    Ramen = "ramen"
    Mustard = "mustard"
    Milk = "milk"
    Gelatin = "gelatin"
    Cracker = "cracker"
    Oat = "oat"
    Cereal = "cereal"
    Biscuit = "biscuit"


def get_tml_asset_path(object_str):
    return os.path.join(tml_root, object_str, "OBJ/3DModel.obj")


def create_tml_asset(gym, sim, object_str, asset_options=None):
    if asset_options is None:
        asset_options = gymapi.AssetOptions()
        asset_options.density = 1e3
        asset_options.override_inertia = True
        asset_options.angular_damping = 0.1
        asset_options.linear_damping = 0.1
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.resolution = 300000
        asset_options.vhacd_params.max_convex_hulls = 10
        asset_options.vhacd_params.max_num_vertices_per_ch = 64
    asset = load_obj(
        gym,
        sim,
        os.path.join(tml_root, object_str),
        "OBJ/3DModel.obj",
        1.0,
        asset_options,
    )
    return asset
