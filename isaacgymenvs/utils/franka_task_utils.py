import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.amp.poselib.poselib.core.rotation3d import *
from isaacgymenvs.utils.asset_utils import *

from scipy.spatial.transform import Rotation as scipy_rot

# These should match franka_table_scene.yaml in contact_demo
table_pos = [0.0, 0.0, 1.0]
table_thickness = 0.05  # The tabletop height is 1.025
table_xy_dims = [1.0, 1.2]
# Create table stand asset
table_stand_height = 0.013  # 0.5in on hardware
table_stand_pos = [
    -0.5,
    0.0,
    1.0 + table_thickness / 2 + table_stand_height / 2,
]
# Define start pose for franka
franka_start_pose = gymapi.Transform()
franka_start_pose.p = gymapi.Vec3(
    -0.45, 0.0, 1.0 + table_thickness / 2 + table_stand_height
)
franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

# Define start pose for table
table_start_pose = gymapi.Transform()
table_start_pose.p = gymapi.Vec3(*table_pos)
table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

# Define start pose for table stand
table_stand_start_pose = gymapi.Transform()
table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
# Robot world transform
X_WR = np.eye(4)
X_WR[:3, 3] = np.array(
    [franka_start_pose.p.x, franka_start_pose.p.y, franka_start_pose.p.z]
)
X_WR[:3, :3] = scipy_rot.from_quat(
    [
        franka_start_pose.r.x,
        franka_start_pose.r.y,
        franka_start_pose.r.z,
        franka_start_pose.r.w,
    ]
).as_matrix()
wall_size = [0.045, 1.015, 0.1]


def compute_wall_yaw_offset(left_x, right_x, wall_width):
    """
    Given the x positions of the left and right ends of the wall,
    compute the position and orientation of the wall
    left and right x positions are relative to the robot (+x)
    """
    return np.arctan2(right_x - left_x, wall_width)


def create_franka_wall_envs(
    task, num_envs, spacing, num_per_row, disable_cubeB_gravity=False
):
    lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    upper = gymapi.Vec3(spacing, spacing, spacing)

    asset_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../assets"
    )
    franka_asset_file = "urdf/franka_description/robots/franka_panda_gripper.urdf"

    if "asset" in task.cfg["env"]:
        asset_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            task.cfg["env"]["asset"].get("assetRoot", asset_root),
        )
        franka_asset_file = task.cfg["env"]["asset"].get(
            "assetFileNameFranka", franka_asset_file
        )
    ############################

    # this should be updated

    yaw = compute_wall_yaw_offset(
        task.cfg["env"]["wall_left_x"], task.cfg["env"]["wall_right_x"], wall_size[1]
    )
    R_wall = scipy_rot.from_euler("z", yaw).as_matrix()
    # Compute the position of the wall
    wall_center_shift = R_wall @ np.array([wall_size[0] / 2.0, 0, 0])
    wall_pose_default = np.array(
        [
            table_xy_dims[0] / 2 + wall_size[0] / 2.0,
            0.0,
            0.5 * wall_size[2] + 1.0 + table_thickness / 2,
        ]
    )
    wall_pose_default[0] = (
        np.average([task.cfg["env"]["wall_left_x"], task.cfg["env"]["wall_right_x"]])
        + franka_start_pose.p.x
    )

    wall_pose_default += wall_center_shift

    wall_pose = gymapi.Transform()
    wall_pose.p = gymapi.Vec3(*wall_pose_default)
    wall_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), yaw)

    ############################

    # load franka asset
    asset_options = gymapi.AssetOptions()
    asset_options.flip_visual_attachments = True
    asset_options.fix_base_link = True
    asset_options.collapse_fixed_joints = False
    asset_options.disable_gravity = True
    asset_options.thickness = 0.001
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
    asset_options.use_mesh_materials = True
    franka_asset = task.gym.load_asset(
        task.sim, asset_root, franka_asset_file, asset_options
    )
    # Set finger drive mode to effort
    franka_dof_stiffness = to_torch(
        [0, 0, 0, 0, 0, 0, 0, 5000.0, 5000.0], dtype=torch.float, device=task.device
    )
    franka_dof_damping = to_torch(
        [0, 0, 0, 0, 0, 0, 0, 1.0e2, 1.0e2], dtype=torch.float, device=task.device
    )

    table_opts = gymapi.AssetOptions()
    table_opts.fix_base_link = True
    table_asset = task.gym.create_box(
        task.sim, table_xy_dims[0], table_xy_dims[1], table_thickness, table_opts
    )

    table_stand_opts = gymapi.AssetOptions()
    table_stand_opts.fix_base_link = True
    table_stand_asset = task.gym.create_box(
        task.sim, *[0.2, 0.2, table_stand_height], table_opts
    )
    # Create wall asset
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    wall_asset = task.gym.create_box(
        task.sim, wall_size[0], wall_size[1], wall_size[2], asset_options
    )
    task.cubeA_size = 0.1
    task.cubeB_size = 0.1
    # Create cubeA asset
    # cubeA_opts = gymapi.AssetOptions()
    # cubeA_asset = self.gym.create_box(
    #     self.sim, *([self.cubeA_size] * 3), cubeA_opts
    # )
    # cubeA_color = gymapi.Vec3(0.6, 0.1, 0.0)

    # # Create cubeB asset
    # cubeB_opts = gymapi.AssetOptions()
    # cubeB_asset = self.gym.create_box(
    #     self.sim, *([self.cubeB_size] * 3), cubeB_opts
    # )
    # cubeB_color = gymapi.Vec3(0.0, 0.4, 0.1)

    cubeA_color = gymapi.Vec3(0.6, 0.1, 0.0)  # Red
    cubeB_color = gymapi.Vec3(0.0, 0.4, 0.1)  # Green

    task.num_franka_bodies = task.gym.get_asset_rigid_body_count(franka_asset)
    task.num_franka_dofs = task.gym.get_asset_dof_count(franka_asset)

    print("num franka bodies: ", task.num_franka_bodies)
    print("num franka dofs: ", task.num_franka_dofs)

    # set franka dof properties
    franka_dof_props = task.gym.get_asset_dof_properties(franka_asset)
    task.franka_dof_lower_limits = []
    task.franka_dof_upper_limits = []
    task._franka_effort_limits = []
    for i in range(task.num_franka_dofs):
        franka_dof_props["driveMode"][i] = gymapi.DOF_MODE_EFFORT
        # franka_dof_props["driveMode"][i] = (
        #     gymapi.DOF_MODE_POS if i > 6 else gymapi.DOF_MODE_EFFORT
        # )
        # if task.physics_engine == gymapi.SIM_PHYSX:
        #     franka_dof_props["stiffness"][i] = franka_dof_stiffness[i]
        #     franka_dof_props["damping"][i] = franka_dof_damping[i]
        # else:
        #     franka_dof_props["stiffness"][i] = 7000.0
        #     franka_dof_props["damping"][i] = 50.0

        task.franka_dof_lower_limits.append(franka_dof_props["lower"][i])
        task.franka_dof_upper_limits.append(franka_dof_props["upper"][i])
        task._franka_effort_limits.append(franka_dof_props["effort"][i])

    # Set gripper finger drive mode to POS
    franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
    franka_dof_props["stiffness"][7:].fill(800.0)
    franka_dof_props["damping"][7:].fill(40.0)

    task.franka_dof_lower_limits = to_torch(
        task.franka_dof_lower_limits, device=task.device
    )
    task.franka_dof_upper_limits = to_torch(
        task.franka_dof_upper_limits, device=task.device
    )
    task._franka_effort_limits = to_torch(
        task._franka_effort_limits, device=task.device
    )
    task.franka_dof_speed_scales = torch.ones_like(task.franka_dof_lower_limits)
    task.franka_dof_speed_scales[[7, 8]] = 0.1
    franka_dof_props["effort"][7] = 200
    franka_dof_props["effort"][8] = 200

    task._table_surface_pos = np.array(table_pos) + np.array(
        [0, 0, table_thickness / 2]
    )
    task.reward_settings["table_height"] = task._table_surface_pos[2]

    # Define start pose for table stand
    table_stand_start_pose = gymapi.Transform()
    table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
    table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    # Define start pose for cubes (doesn't really matter since they're get overridden during reset() anyways)
    cubeA_start_pose = gymapi.Transform()
    cubeA_start_pose.p = gymapi.Vec3(-1.0, 0.0, 0.0)
    cubeA_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    cubeB_start_pose = gymapi.Transform()
    cubeB_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
    cubeB_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    task.chosen_objs = (
        np.array(task.cfg["env"]["chosen_objs"])
        if task.cfg["run_cfg"]["hardware_object"] is None
        else np.array([task.cfg["run_cfg"]["hardware_object"]])
    )

    # compute aggregate size
    num_franka_bodies = task.gym.get_asset_rigid_body_count(franka_asset)
    num_franka_shapes = task.gym.get_asset_rigid_shape_count(franka_asset)
    max_agg_bodies = (
        num_franka_bodies + 5
    )  # 1 for table, table stand, cubeA, cubeB, wall
    max_agg_shapes = num_franka_shapes + 4 + len(task.chosen_objs)

    # Create environments
    obj_assets = []
    obj_assets_viz = []

    task.frankas = []
    task.envs = []

    # self.init_latent()
    # make the objects ligihter?
    asset_options = gymapi.AssetOptions()
    asset_options.density = 1e3
    asset_options.override_inertia = True
    asset_options.angular_damping = 0.1
    asset_options.linear_damping = 0.1
    asset_options.vhacd_enabled = True
    asset_options.vhacd_params.resolution = 300000
    asset_options.vhacd_params.max_convex_hulls = 10
    asset_options.vhacd_params.max_num_vertices_per_ch = 64

    asset_options_viz = gymapi.AssetOptions()
    asset_options_viz.density = 1e5
    asset_options_viz.angular_damping = 100.0
    asset_options_viz.linear_damping = 100.0
    asset_options_viz.max_angular_velocity = 0.01
    asset_options_viz.max_linear_velocity = 0.01
    asset_options_viz.disable_gravity = disable_cubeB_gravity

    for obj in task.chosen_objs:
        obj_assets.append(
            create_tml_asset(
                task.gym,
                task.sim,
                obj,
                asset_options=asset_options,
            )
        )
        obj_assets_viz.append(
            create_tml_asset(
                task.gym,
                task.sim,
                obj,
                asset_options=asset_options_viz,
            )
        )
    for i in range(task.num_envs):
        # create env instance
        env_ptr = task.gym.create_env(task.sim, lower, upper, num_per_row)

        # Create actors and define aggregate group appropriately depending on setting
        # NOTE: franka should ALWAYS be loaded first in sim!
        if task.aggregate_mode >= 3:
            task.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

        # Create franka
        # Potentially randomize start pose
        if task.franka_position_noise > 0:
            rand_xyz = task.franka_position_noise * (-1.0 + np.random.rand(3) * 2.0)
            franka_start_pose.p = gymapi.Vec3(
                -0.45 + rand_xyz[0],
                0.0 + rand_xyz[1],
                1.0 + table_thickness / 2 + table_stand_height + rand_xyz[2],
            )
        if task.franka_rotation_noise > 0:
            rand_rot = torch.zeros(1, 3)
            rand_rot[:, -1] = task.franka_rotation_noise * (
                -1.0 + np.random.rand() * 2.0
            )
            new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
            franka_start_pose.r = gymapi.Quat(*new_quat)
        franka_actor = task.gym.create_actor(
            env_ptr,
            franka_asset,
            franka_start_pose,
            "franka",
            i,
            0b10,
            0,  # Franka should not collide with cube B
        )
        task.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

        if task.aggregate_mode == 2:
            task.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

        # Create table
        table_actor = task.gym.create_actor(
            env_ptr, table_asset, table_start_pose, "table", i, 0b00, 0
        )
        table_stand_actor = task.gym.create_actor(
            env_ptr,
            table_stand_asset,
            table_stand_start_pose,
            "table_stand",
            i,
            0b01,
            0,
        )

        # Create wall
        wall_actor = task.gym.create_actor(
            env_ptr, wall_asset, wall_pose, "wall", i, 0b00, 0
        )

        if task.aggregate_mode == 1:
            task.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

        # Create cubes
        # self._cubeA_id = self.gym.create_actor(
        #     env_ptr, cubeA_asset, cubeA_start_pose, "cubeA", i, 2, 0  # 010
        # )
        task._cubeA_id = task.gym.create_actor(
            env_ptr,
            obj_assets[i % len(task.chosen_objs)],
            cubeA_start_pose,
            "cubeA",
            i,
            0b01,  # Cube a
            0,
        )
        task._cubeB_id = task.gym.create_actor(
            env_ptr,
            obj_assets_viz[i % len(task.chosen_objs)],
            cubeB_start_pose,
            "cubeB",
            i,
            0b11,  # This should collide with the table and the wall only
            0,  # 110
        )
        # Set colors
        # task.gym.set_rigid_body_color(
        #     env_ptr, task._cubeA_id, 0, gymapi.MESH_VISUAL, cubeA_color
        # )
        # task.gym.set_rigid_body_color(
        #     env_ptr, task._cubeB_id, 0, gymapi.MESH_VISUAL, cubeB_color
        # )

        if task.aggregate_mode > 0:
            task.gym.end_aggregate(env_ptr)

        # Store the created env pointers
        task.envs.append(env_ptr)
        task.frankas.append(franka_actor)

        # # Update the latent based on the object
        # # TODO: this is a hack, fix this
        # self.states["cubeA_latent"][i, 0] = i % len()
        # self.states["cubeB_latent"][i, 0] = i % len()
