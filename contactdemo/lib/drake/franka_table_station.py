import numpy as np
import pydot
import open3d as o3d
import contactdemo.lib.utils.math_utils as math_utils
import pydrake.solvers as mp
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    JointStiffnessController,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    MultibodyPlant,
    Parser,
    Simulator,
    StartMeshcat,
    FindResourceOrThrow,
    RigidTransform,
    ConstantVectorSource,
    SpatialInertia,
    UnitInertia,
    CoulombFriction,
)

from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.multibody.tree import FixedOffsetFrame
from pydrake.math import RotationMatrix, RollPitchYaw
from pydrake.common.eigen_geometry import Quaternion
import os
from enum import Enum
import copy

pwd = os.path.dirname(os.path.realpath(__file__))
asset_root = os.path.join(pwd, "../../assets")


def drake_mesh_object_setup_fn(
    drake_hand_plant, X_WO, drake_shape, o3d_mesh, diffuse_color=[1, 1, 1, 0.7]
):
    import open3d as o3d

    mu_static = 0.0
    mu_dynamic = 0.0

    obj_model_instance = drake_hand_plant.plant.AddModelInstance("obj_model_instance")
    spatial_inertia = SpatialInertia(
        mass=1.0, p_PScm_E=np.zeros(3), G_SP_E=UnitInertia(0.0, 0.0, 0.0)
    )
    drake_hand_plant.object_body = drake_hand_plant.plant.AddRigidBody(
        name="obj", M_BBo_B=spatial_inertia, model_instance=obj_model_instance
    )
    body_X_BG = RigidTransform([0.0, 0.0, 0.0])
    body_friction = CoulombFriction(
        static_friction=mu_static, dynamic_friction=mu_dynamic
    )
    drake_hand_plant.plant.RegisterVisualGeometry(
        body=drake_hand_plant.object_body,
        X_BG=body_X_BG,
        shape=drake_shape,
        name="obj_visual",
        diffuse_color=diffuse_color,
    )
    drake_hand_plant.plant.RegisterCollisionGeometry(
        body=drake_hand_plant.object_body,
        X_BG=body_X_BG,
        shape=drake_shape,
        name="obj_collision",
        coulomb_friction=body_friction,
    )
    drake_hand_plant.object_frame = drake_hand_plant.object_body.body_frame()
    # Tabletop is at xy plane
    drake_hand_plant.plant.WeldFrames(
        drake_hand_plant.plant.world_frame(), drake_hand_plant.object_frame, X_WO
    )
    # Add the o3d object
    drake_hand_plant.o3d_mesh = copy.deepcopy(o3d_mesh).transform(X_WO.GetAsMatrix4())
    drake_hand_plant.o3d_mesh.compute_triangle_normals()
    drake_hand_plant.o3d_mesh.compute_vertex_normals()
    assert (
        drake_hand_plant.o3d_mesh.has_triangle_normals()
        and drake_hand_plant.o3d_mesh.has_vertex_normals()
    )
    # assert drake_hand_plant.o3d_mesh.is_watertight()
    drake_hand_plant.o3d_scene = o3d.t.geometry.RaycastingScene()
    drake_hand_plant.o3d_scene.add_triangles(
        o3d.t.geometry.TriangleMesh.from_legacy(drake_hand_plant.o3d_mesh)
    )


class RobotFinger(Enum):
    LEFT = 0
    RIGHT = 1


class FrankaTableSceneObject(Enum):
    TABLE = 0
    WALL = 1
    ROBOT = 2
    OBSTACLES = 3


fingertip_center_offset = np.array([0.0, 0.0, 0.032])


class FrankaTableStation:
    def __init__(self, cfg, object_file=None) -> None:
        """
        Creates a MultibodyPlant with a table and a Franka Panda arm
        """
        self.builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            self.builder, time_step=1e-4
        )
        self.parser = Parser(self.plant, self.scene_graph)
        # TODO: make sure drake's default franka model matches the Isaac Gym one
        self.franka_model = self.parser.AddModelsFromUrl(
            "package://drake/manipulation/models/franka_description/urdf/panda_arm_hand.urdf"
        )[0]
        self.table_model = self.parser.AddModelFromFile(
            # FindResourceOrThrow(
            os.path.join(
                asset_root,
                "primitive_shapes/table.urdf",
            )
        )
        self.table_stand_model = self.parser.AddModelFromFile(
            # FindResourceOrThrow(
            os.path.join(
                asset_root,
                "primitive_shapes/table_stand.urdf",
            )
            # )
        )
        self.wall_model = self.parser.AddModelFromFile(
            # FindResourceOrThrow(
            os.path.join(
                asset_root,
                "primitive_shapes/wall.urdf",
            )
        )

        # Add obstacles
        self.obstacles = []
        if "obstacles" in cfg.keys():
            for obstacle in cfg["obstacles"].keys():
                obstacle_file = cfg["obstacles"][obstacle]["file"]
                self.obstacles.append(
                    self.parser.AddModelFromFile(
                        os.path.join(asset_root, obstacle_file)
                    )
                )

        # Add the object
        # FIXME: cleaner implementation
        if object_file is not None:
            object_body_name = object_file.split("/")[-1].split(".")[0]
            self.object_model = self.parser.AddModelFromFile(os.path.join(object_file))
        self.cfg = cfg
        # # weld the frames
        self.plant.WeldFrames(
            self.plant.world_frame(),
            self.plant.GetFrameByName("table_base_link"),
            X_FM=RigidTransform(p=self.cfg["table"]["table_pos"]),
        )
        self.plant.WeldFrames(
            self.plant.world_frame(),
            self.plant.GetFrameByName("table_stand_base_link"),
            X_FM=RigidTransform(p=self.cfg["table_stand"]["table_stand_pos"]),
        )
        self.plant.WeldFrames(
            self.plant.world_frame(),
            self.plant.GetFrameByName("wall_base_link"),
            X_FM=RigidTransform(
                p=self.cfg["wall"]["wall_pos"],
                rpy=RollPitchYaw(self.cfg["wall"]["wall_rpy"]),
            ),
        )
        self.plant.WeldFrames(
            self.plant.world_frame(),
            self.plant.GetFrameByName("panda_link0"),
            X_FM=RigidTransform(p=self.cfg["franka"]["franka_pos"]),
        )

        # Weld the obstacles
        self.obstacle_names = []
        if self.obstacles:
            for obs in cfg["obstacles"].keys():
                obs_cfg = self.cfg["obstacles"][obs]
                self.obstacle_names.append(obs_cfg["base_link_name"])
                self.plant.WeldFrames(
                    self.plant.world_frame(),
                    self.plant.GetFrameByName(obs_cfg["base_link_name"]),
                    X_FM=RigidTransform(
                        p=obs_cfg["pos"], quaternion=Quaternion(obs_cfg["quat"])
                    ),
                )

        # Weld the object to the table
        # plant.WeldFrames(
        #     plant.GetFrameByName("table_base_link"),
        #     plant.GetFrameByName("base_link_cracker"),
        #     X_FM=RigidTransform(),
        # )

        self.plant.Finalize()
        self.inspector = self.scene_graph.model_inspector()

        # Convenience variables
        self.tabletop_frame_id = self.plant.GetFrameByName("tabletop_frame")
        self.table_body = self.plant.GetBodyByName("table_base_link")
        if object_file is not None:
            self.object_body = self.plant.GetBodyByName(object_body_name)
        self.franka_hand_body = self.plant.GetBodyByName("panda_hand")
        self.wall_body = self.plant.GetBodyByName("wall_base_link")
        self.wall_bottom_edge_frame_id = self.plant.GetFrameByName(
            "wall_bottom_edge_frame"
        )
        self.robot_finger_bodies = dict()
        self.robot_finger_bodies[RobotFinger.LEFT] = self.plant.GetBodyByName(
            "panda_leftfinger"
        )
        self.robot_finger_bodies[RobotFinger.RIGHT] = self.plant.GetBodyByName(
            "panda_rightfinger"
        )
        self.obstacle_bodies = []
        if self.obstacles:
            for obs_name in self.obstacle_names:
                self.obstacle_bodies.append(self.plant.GetBodyByName(obs_name))
        # TODO: add frame for point between fingers
        self.flange_to_finger_center_offset = np.array(
            [0.0, 0.0, 0.1034]
        )  # https://download.franka.de/documents/220010_Product%20Manual_Franka%20Hand_1.2_EN.pdf
        if self.cfg["visualize"]:
            # Create visualizer
            self.meshcat = StartMeshcat()
            self.visualizer = MeshcatVisualizer.AddToBuilder(
                self.builder, self.scene_graph, self.meshcat
            )
        else:
            self.visualizer = None
        # Add the actuators
        # TODO replace the simple PD controller with something more sophisticated
        self.franka_dof = (
            self.plant.get_state_output_port(self.franka_model).size() // 2
        )
        # Hack for setting zero torque for everything
        self.torques_system = self.builder.AddSystem(
            ConstantVectorSource(np.zeros(self.plant.num_actuators()))
        )
        self.builder.Connect(
            self.torques_system.get_output_port(), self.plant.get_actuation_input_port()
        )

        self.diagram = self.builder.Build()
        self.X_WR = np.eye(4)  # We assume the franka is oriented toward +x
        self.X_WR[:3, 3] = self.cfg["franka"]["franka_pos"]

        # Create an Open3D model of the object and compute the bounding box
        if object_file is not None:
            self.obj_o3d_mesh = o3d.io.read_triangle_mesh(object_file)
            # FIXME: this is a hack for better bounding box quality
            self.obj_o3d_bbox = (
                o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(
                    self.obj_o3d_mesh.get_axis_aligned_bounding_box()
                )
            )
            self.obj_o3d_bbox.color = [1, 0, 0]

            # Compute the vertices of the bounding box
            self.obj_o3d_bbox_vertices = np.asarray(self.obj_o3d_bbox.get_box_points())

        diagram_context = self.create_default_diagram_context()
        plant_context = self.diagram.GetMutableSubsystemContext(
            self.plant, diagram_context
        )
        self.X_wall_bottom_edge = self.wall_bottom_edge_frame_id.CalcPoseInWorld(
            plant_context
        )
        # Note that this is defined as the -x direction of the frame
        self.wall_xy_normal = -self.X_wall_bottom_edge.rotation().multiply([1, 0, 0])[
            :2
        ]
        # normalize
        self.wall_xy_normal /= np.linalg.norm(self.wall_xy_normal)
        self.wall_normal = np.array([self.wall_xy_normal[0], self.wall_xy_normal[1], 0])

        self.table_dims = np.array(self.cfg["table"]["table_dims"])
        self.table_pos = np.array(self.cfg["table"]["table_pos"])
        self.world_lb = self.table_pos.copy()
        self.world_lb[:2] -= self.table_dims[:2] / 2
        self.world_lb[2] += self.table_dims[2] / 2
        self.world_ub = self.table_pos.copy()
        self.world_ub[:2] += self.table_dims[:2] / 2
        self.world_ub[2] = np.infty
        # clip the ub with wall
        self.world_ub[0] = min(
            self.world_ub[0], self.X_wall_bottom_edge.translation()[0]
        )

    def point_to_wall_xy_distance(self, p_W):
        p_W = np.atleast_2d(p_W)
        # Compute the distance between the point and the wall in the xy plane
        p_W = p_W[:, :2]
        wall_center = self.X_wall_bottom_edge.translation()[:2]
        return np.dot(p_W - wall_center, self.wall_xy_normal)

    def run_simulation(self, duration, diagram_context=None):
        if diagram_context is None:
            diagram_context = self.diagram.CreateDefaultContext()
        simulator = Simulator(self.diagram, diagram_context)
        simulator.set_target_realtime_rate(1.0)
        self.visualizer.StartRecording()

        simulator.AdvanceTo(duration)

        self.visualizer.StopRecording()
        self.visualizer.PublishRecording()

    def create_default_diagram_context(self):
        return self.diagram.CreateDefaultContext()

    def create_default_diagram_and_plant_context(self):
        diagram_context = self.create_default_diagram_context()
        plant_context = self.diagram.GetMutableSubsystemContext(
            self.plant, diagram_context
        )
        return diagram_context, plant_context

    def set_positions(self, positions, diagram_context=None):
        if diagram_context is None:
            diagram_context = self.create_default_diagram_context()
        plant_context = self.diagram.GetMutableSubsystemContext(
            self.plant, diagram_context
        )
        self.plant.SetPositions(plant_context, positions)
        return diagram_context

    def add_object_env_distance_constraint_to_ik(
        self, ik, distance_lower, distance_upper, env_obj
    ):
        """
        Construct a constraint on the distance between the manipuland and an environment object
        :param ik: The inverse kinematics object
        :param distance_lower: The lower bound on the distance between the object and the env_obj
        :param distance_upper: The upper bound on the distance between the object and the env_obj
        :param env_obj: The environment object to which the manipuland is in contact with
        """
        assert distance_lower <= distance_upper
        constraints = []
        if env_obj == FrankaTableSceneObject.TABLE:
            constraints = ik.AddDistanceConstraint(
                geometry_pair=(
                    self.plant.GetCollisionGeometriesForBody(self.object_body)[0],
                    self.plant.GetCollisionGeometriesForBody(self.table_body)[0],
                ),
                distance_lower=distance_lower,
                distance_upper=distance_upper,
            )
        elif env_obj == FrankaTableSceneObject.WALL:
            constraints = ik.AddDistanceConstraint(
                geometry_pair=(
                    self.plant.GetCollisionGeometriesForBody(self.object_body)[0],
                    self.plant.GetCollisionGeometriesForBody(self.wall_body)[0],
                ),
                distance_lower=distance_lower,
                distance_upper=distance_upper,
            )
        elif env_obj == FrankaTableSceneObject.OBSTACLES:
            for obs in self.obstacle_bodies:
                constraints.append(
                    ik.AddDistanceConstraint(
                        geometry_pair=(
                            self.plant.GetCollisionGeometriesForBody(self.object_body)[
                                0
                            ],
                            self.plant.GetCollisionGeometriesForBody(obs)[0],
                        ),
                        distance_lower=distance_lower,
                        distance_upper=distance_upper,
                    )
                )
        else:
            raise ValueError("env_obj must be either 'table' or 'wall'")
        return constraints

    def add_object_pair_minimum_distance_constraint_to_ik(
        self, ik, min_distance, influence_distance_offset=0.01
    ):
        constraint = ik.AddMinimumDistanceLowerBoundConstraint(
            bound=min_distance,
            influence_distance_offset=influence_distance_offset,
        )
        return constraint

    def add_obj_point_to_env_point_distance_constraint_to_ik(
        self, ik, p_BC, p_EC, distance_lower, distance_upper, env_obj
    ):
        """
        Construct a constraint on the distance between a point on the manipuland
        to a point on an environment object
        :param ik: The inverse kinematics object
        :param p_BC: The point on the manipuland in manipuland frame
        :param p_EC: The point on the environment object in environment object frame
        :param distance_lower: The lower bound on the distance between the point and the env_obj
        :param distance_upper: The upper bound on the distance between the point and the env_obj
        :param env_obj: The environment object to which the manipuland is in contact with
        """
        assert distance_lower <= distance_upper
        if env_obj == "table":
            constraint = ik.AddPointToPointDistanceConstraint(
                frame1=self.object_body.body_frame(),
                p_B1P1=p_BC,
                frame2=self.tabletop_frame_id,
                p_B2P2=p_EC,
                distance_lower=distance_lower,
                distance_upper=distance_upper,
            )
        elif env_obj == "wall":
            constraint = ik.AddPointToPointDistanceConstraint(
                frame1=self.object_body.body_frame(),
                p_B1P1=p_BC,
                frame2=self.wall_bottom_edge_frame_id,
                p_B2P2=p_EC,
                distance_lower=distance_lower,
                distance_upper=distance_upper,
            )
        elif env_obj == "world":
            constraint = ik.AddPointToPointDistanceConstraint(
                frame1=self.object_body.body_frame(),
                p_B1P1=p_BC,
                frame2=self.plant.world_frame(),
                p_B2P2=p_EC,
                distance_lower=distance_lower,
                distance_upper=distance_upper,
            )
        return constraint

    def add_obj_point_to_wall_bottom_dist_constraint_to_ik(
        self, ik, p_BC, distance_lower, distance_upper
    ):
        assert distance_lower <= distance_upper
        constraint = ik.AddPointToLineDistanceConstraint(
            self.object_body.body_frame(),
            p_BC,
            self.wall_bottom_edge_frame_id,
            np.zeros(3),  # the line is the y axis of wall_bottom_edge_frame
            np.array([0, 1, 0]),
            distance_lower,
            distance_upper,
        )
        return constraint

    def add_obj_env_contact_constraints_to_ik(
        self,
        ik,
        p_BC,
        p_EC,
        contact_points_dist_lower,
        contact_points_dist_upper,
        object_pair_dist_lower,
        object_pair_dist_upper,
        env_obj,
    ):
        """
        Construct the two constraints for the object to be in contact with an environment object
        1. p_BC and p_EC to be bounded by contact_points_dist
        2. Distance between object B and environment object E to be bounded by object_pair_dist
        and distance_upper
        :param ik: The inverse kinematics object
        :param p_BC: The point on the object in object frame
        :param p_EC: The point on the environment object in environment object frame
        :param contact_points_dist_lower: The lower bound on the distance between p_BC and p_EC
        :param contact_points_dist_upper: The upper bound on the distance between p_BC and p_EC
        :param object_pair_dist_lower: The lower bound on the distance between the object and the env_obj
        :param object_pair_dist_upper: The upper bound on the distance between the object and the env_obj given by
        :param env_obj: The environment object to which the object is in contact with
        """
        constraints = []
        constraints.append(
            self.add_obj_point_to_env_point_distance_constraint_to_ik(
                ik,
                p_BC,
                p_EC,
                contact_points_dist_lower,
                contact_points_dist_upper,
                env_obj,
            )
        )
        constraints.append(
            self.add_object_env_distance_constraint_to_ik(
                ik, object_pair_dist_lower, object_pair_dist_upper, env_obj
            )
        )
        return constraints

    def add_obj_bounding_box_above_table_constraint(self, ik, tol=5e-3):
        """
        Add a constraint to the inverse kinematics problem that the object is above the table
        """
        lb = -self.table_dims / 2  # lower corner of table
        ub = self.table_dims / 2  # upper corner of table
        lb[-1] = -0.005  # z coordinate
        ub[-1] = np.infty  # z coordinate
        constraints = []  # xyz
        for vert in self.obj_o3d_bbox_vertices:
            constraints.append(
                ik.AddPositionConstraint(
                    frameB=self.object_body.body_frame(),
                    p_BQ=vert,
                    frameA=self.tabletop_frame_id,
                    p_AQ_lower=lb,
                    p_AQ_upper=ub,
                )
            )
        return constraints

    def add_obj_q_constraint(self, ik, q_desired, tol=1e-3):
        "constraint the object q"
        assert np.prod(q_desired.shape) == 7
        prog = ik.get_mutable_prog()
        pose_WB = self.get_quat_pos_WB(ik.q())
        A = np.eye(7)
        constraint = prog.AddLinearConstraint(
            A=A,
            lb=q_desired - tol,
            ub=q_desired + tol,
            vars=pose_WB,
        )
        return constraint

    def add_obj_R_WB_constraint(self, ik, R_WB, theta_bound=0.01):
        """
        :param ik: The inverse kinematics object
        :param R_WB: The rotation matrix from world frame to object body frame
        :param theta_bound: The bound on the angle between the object frame and the world frame
        """
        constraint = ik.AddOrientationConstraint(
            frameAbar=self.plant.world_frame(),
            R_AbarA=RotationMatrix(R_WB),
            frameBbar=self.object_body.body_frame(),
            R_BbarB=RotationMatrix(),
            theta_bound=theta_bound,
        )
        return constraint

    def add_hand_R_WH_constraint(self, ik, R_WH, theta_bound=0.01):
        """
        The hand orientation in Drake follows the Franka convention,
        x = front of hand
        y = connector to button
        z = finger extensions direction
        :param ik: The inverse kinematics object
        :param R_WH: The rotation matrix from world frame to hand frame
        :param theta_bound: The bound on the angle between the hand frame and the world frame
        """

        constraint = ik.AddOrientationConstraint(
            frameAbar=self.plant.world_frame(),
            R_AbarA=RotationMatrix(R_WH),
            frameBbar=self.franka_hand_body.body_frame(),
            R_BbarB=RotationMatrix(),
            theta_bound=theta_bound,
        )
        return constraint

    def add_fingertip_to_object_distance_constraint_to_ik(
        self, ik, finger_type, distance_lower, distance_upper
    ):
        """
        :param ik: The inverse kinematics object
        :param p_BC: The point on the object in object frame
        :param finger_type: The finger type, either RobotFinger.LEFT or RobotFinger.RIGHT
        :param distance_lower: The lower bound on the distance between the point and the finger
        :param distance_upper: The upper bound on the distance between the point and the finger
        """
        assert distance_lower <= distance_upper
        constraint = ik.AddDistanceConstraint(
            geometry_pair=(
                self.plant.GetCollisionGeometriesForBody(self.object_body)[0],
                self.plant.GetCollisionGeometriesForBody(
                    self.robot_finger_bodies[finger_type]
                )[0],
            ),
            distance_lower=distance_lower,
            distance_upper=distance_upper,
        )
        return constraint

    def add_franka_finger_gap_constraint_to_ik(self, ik, min_gap=0.0, max_gap=0.04):
        """
        Constrain how open the gripper can be. Note that
        """
        assert min_gap >= 0.0 and max_gap <= 0.04  # gap range for each finger
        gripper_q = self.get_franka_joint_angles(ik.q())[-2:]
        A = np.eye(2)
        constraint = ik.get_mutable_prog().AddLinearConstraint(
            A=A,
            lb=min_gap * np.ones((2, 1)),
            ub=max_gap * np.ones((2, 1)),
            vars=gripper_q,
        )
        return constraint

    def add_franka_hand_to_object_distance_constraint_to_ik(
        self, ik, distance_lower, distance_upper
    ):
        """
        :param ik: The inverse kinematics object
        :param distance_lower: The lower bound on the distance between the point and the finger
        :param distance_upper: The upper bound on the distance between the point and the finger
        """
        assert distance_lower <= distance_upper
        constraint = ik.AddDistanceConstraint(
            geometry_pair=(
                self.plant.GetCollisionGeometriesForBody(self.object_body)[0],
                self.plant.GetCollisionGeometriesForBody(self.franka_hand_body)[0],
            ),
            distance_lower=distance_lower,
            distance_upper=distance_upper,
        )
        return constraint

    def add_fingertip_to_obj_p_BC_distance_constraint_to_ik(
        self, ik, finger_type, p_BC, distance_lower, distance_upper
    ):
        """
        :param ik: The inverse kinematics object
        :param p_BC: The point on the object in object frame
        :param finger_type: The finger type, either RobotFinger.LEFT or RobotFinger.RIGHT
        :param distance_lower: The lower bound on the distance between the point and the finger
        :param distance_upper: The upper bound on the distance between the point and the finger
        """
        assert distance_lower <= distance_upper
        constraint = ik.AddPointToPointDistanceConstraint(
            frame1=self.object_body.body_frame(),
            p_B1P1=p_BC,
            frame2=self.robot_finger_bodies[finger_type].body_frame(),
            p_B2P2=fingertip_center_offset,
            distance_lower=distance_lower,
            distance_upper=distance_upper,
        )
        return constraint

    def add_fingertip_center_to_p_W_constraint_to_ik(self, ik, p_W_lower, p_W_upper):
        """
        Constraint the fingertip center to be at p_W with Linf tolerance tol
        """
        assert np.all(p_W_lower <= p_W_upper)
        constraint = ik.AddPositionConstraint(
            frameB=self.plant.GetBodyByName("panda_hand").body_frame(),
            p_BQ=self.flange_to_finger_center_offset,
            frameA=self.plant.world_frame(),
            p_AQ_lower=p_W_lower,
            p_AQ_upper=p_W_upper,
        )
        return constraint

    def add_obj_p_WB_in_AABB_constraint(self, ik, p_WB_lower, p_WB_upper):
        """
        Constraint the object pose to the AABB defined by p_WB_lower, p_WB_upper
        """
        assert np.all(p_WB_lower <= p_WB_upper)
        constraint = ik.AddPositionConstraint(
            frameB=self.object_body.body_frame(),
            p_BQ=np.zeros(3),
            frameA=self.plant.world_frame(),
            p_AQ_lower=p_WB_lower,
            p_AQ_upper=p_WB_upper,
        )
        return constraint

    def add_arm_q_quadratic_cost(self, ik, arm_q_target, scale=1.0):
        """ """
        Q = np.eye(7) * scale
        cost = ik.get_mutable_prog().AddQuadraticErrorCost(
            Q, arm_q_target, self.get_franka_joint_angles(ik.q())[:7]
        )
        return cost


    def add_obj_p_WB_cost(self, ik, p_WB_target, scale=1.0):
        """
        Add a cost of the form ||p_WB - p_WB_target||^2
        Expanding the quadratic form and dropping the constant yields
        p_WB^T*I*p_WB - 2 p_WB^T*I*p_WB_target
        Thus we have Q = I, c = -2*p_WB_target
        :param ik: The inverse kinematics object
        :param p_WB_target: The target position of the object in the world frame
        :param scale: The scale of the cost
        """
        C = np.eye(3) * scale
        return (
            ik.AddPositionCost(
                self.plant.world_frame(),
                p_WB_target,
                self.object_body.body_frame(),
                [0, 0, 0],
                C,
            ),
        )

    def add_obj_R_WB_cost(self, ik, R_WB_target, scale=1.0):
        """
        Adds a cost of the form scale * (1 - cos(θ)) on the object orientation
        :param ik: The inverse kinematics object
        :param R_WB_target: The rotation matrix from world frame to object body frame
        :param scale: The scale of the cost
        """
        cost = ik.AddOrientationCost(
            frameAbar=self.plant.world_frame(),
            R_AbarA=R_WB_target,
            frameBbar=self.object_body.body_frame(),
            R_BbarB=RotationMatrix(),
            c=scale,
        )
        return cost

    def add_obj_X_WB_cost(
        self, ik, X_WB_target, p_WB_cost_scale=1.0, R_WB_cost_scale=1.0
    ):
        p_WB_cost = self.add_obj_p_WB_quadratic_cost(
            ik, X_WB_target[:3, 3], p_WB_cost_scale
        )
        R_WB_cost = self.add_obj_R_WB_cost(
            ik, RotationMatrix(X_WB_target[:3, :3]), R_WB_cost_scale
        )
        return p_WB_cost, R_WB_cost

    def add_franka_hand_orientaion_from_table_normal_cone_constraint(
        self, ik, theta_bound
    ):
        """
        Add a constraint on the angle between the franka hand frame and the world frame
        Note that the table normal is assumbed to be aligned with the world z
        :param ik: The inverse kinematics object
        :param R_WF: The rotation matrix from world frame to franka hand frame
        :param theta_bound: The bound on the angle between the franka hand frame and the world frame
        """
        assert theta_bound >= 0
        constraint = ik.AddAngleBetweenVectorsConstraint(
            frameA=self.plant.world_frame(),
            na_A=np.array([0, 0, 1.0]).reshape(-1, 1),
            frameB=self.franka_hand_body.body_frame(),
            nb_B=np.array([0, 0, -1]).reshape(-1, 1),
            angle_lower=0.0,
            angle_upper=theta_bound,
        )
        return constraint

    def add_franka_hand_point_forward_normal_cone_constraint(self, ik, theta_bound):
        """
        Add a constraint that the x-axis of the Franka hand frame ("forward plane")
        is within theta from the x-axis of the world frame, which is parallel to the
        x-axis of the franka base frame.
        :param ik: The inverse kinematics object
        :param theta_bound: The bound on the angle between the franka hand frame and the world frame
        """
        assert theta_bound >= 0
        constraint = ik.AddAngleBetweenVectorsConstraint(
            frameA=self.plant.world_frame(),
            na_A=np.array([1.0, 0, 0.0]).reshape(-1, 1),
            frameB=self.franka_hand_body.body_frame(),
            nb_B=np.array([1.0, 0, 0]).reshape(-1, 1),
            angle_lower=0.0,
            angle_upper=theta_bound,
        )
        return constraint

    def add_fingertip_in_wall_normal_gaze_cone_constraint(
        self, ik, finger, p_WB, cone_half_angle
    ):
        """
        Add a constraint where the fingertip position is within a cone a ray starting
        from the a point on the object p_BC and pointing in the direction of the wall normal
        """
        assert cone_half_angle >= 0
        if p_WB is None:
            p_WB = np.zeros(3)
        constraint = ik.AddGazeTargetConstraint(
            frameA=self.plant.world_frame(),
            p_AS=p_WB.reshape(-1, 1),
            n_A=self.wall_normal.reshape(-1, 1),
            frameB=self.robot_finger_bodies[finger].body_frame(),
            p_BT=fingertip_center_offset,
            # https://download.franka.de/documents/220010_Product%20Manual_Franka%20Hand_1.2_EN.pdf
            # FIXME: double check the finger tip position
            cone_half_angle=cone_half_angle,
        )
        return constraint

    def add_fingertip_world_position_constraint(
        self, ik, finger, AABB_lower, AABB_upper
    ):
        """
        Add a constraint on the z coordinate of the fingertip
        """
        assert np.all(AABB_lower <= AABB_upper)
        constraint = ik.AddPositionConstraint(
            frameB=self.robot_finger_bodies[finger].body_frame(),
            p_BQ=fingertip_center_offset,
            frameA=self.plant.world_frame(),
            p_AQ_lower=AABB_lower,
            p_AQ_upper=AABB_upper,
        )
        return constraint

    def add_fingertip_identical_gap_constraint(self, ik):
        """
        Add a constraint that the gap between the two fingers is the same
        """
        A = np.array([[1, -1]])
        return ik.get_mutable_prog().AddLinearEqualityConstraint(
            A,
            0.0,
            FrankaTableStation.get_franka_joint_angles(ik.q())[-2:].reshape(-1, 1),
        )

    def add_object_in_front_of_hand_gaze_constraint(self, ik, cone_half_angle):
        """
        Constraint the object to be within a cone "in front" of the hand,
        i.e. x direction of the hand frame
        https://download.franka.de/documents/220010_Product%20Manual_Franka%20Hand_1.2_EN.pdf
        """
        assert cone_half_angle >= 0
        constraint = ik.AddGazeTargetConstraint(
            frameA=self.franka_hand_body.body_frame(),
            p_AS=self.flange_to_finger_center_offset,
            n_A=np.array([1.0, 0.0, 0.0]).reshape(-1, 1),
            frameB=self.object_body.body_frame(),
            p_BT=np.array([0.0, 0.0, 0.0]).reshape(-1, 1),
            cone_half_angle=cone_half_angle,
        )
        return constraint

    def construct_ik(self, plant_context=None):
        if plant_context is None:
            (
                _,
                plant_context,
            ) = self.create_default_diagram_and_plant_context()

        ik = InverseKinematics(self.plant, plant_context)
        return ik

    # helper functions to index into q and v by object
    @staticmethod
    def get_franka_joint_angles(q):
        """
        Get the franka joint angles
        Franka has 7dof + 2dof fingers
        """
        return q[:9]

    @staticmethod
    def get_p_WB(q):
        """
        Get the object position in the world frame
        the coordinates are qw-qx-qy-qz-x-y-z
        """
        return q[13:16]

    @staticmethod
    def get_quat_WB(q):
        """
        Get the object quaternion in the world frame
        the coordinates are qw-qx-qy-qz-x-y-z
        """
        return q[9:13]

    @staticmethod
    def get_quat_pos_WB(q):
        """
        Get the object pose in the world frame
        the coordinates are qw-qx-qy-qz-x-y-z
        """
        return q[9:16]

    @staticmethod
    def get_X_WB(q):
        """
        Get the object pose in the world frame
        the coordinates are qw-qx-qy-qz-x-y-z
        """
        return math_utils.drake_pose_state_to_X(q[9:16])

    def solve_object_pose_scene_with_contact(
        self,
        p_BC_list,
        p_EC_list,
        contact_object_list,
        X_WB_guess=None,
        franka_q_guess=None,
        return_q=False,
    ):
        """
        Given contact predictions provided by the model, i.e. object-env contacts
        and object-franka contacts, return the full pose of the object.
        :param self: a FrankaTableStation object
        :param p_BC_list: a list of points on the object in object frame
        :param contact_object_list: a list of objects that the object is in contact with
        """
        assert len(p_BC_list) == len(contact_object_list)
        assert len(p_BC_list) == len(p_EC_list)
        # Create IK object
        ik = self.construct_ik()
        # Add contacts
        for p_BC, p_EC, contact_object in zip(
            p_BC_list, p_EC_list, contact_object_list
        ):
            if contact_object == FrankaTableSceneObject.ROBOT:
                # object should be in contact with robot
                raise NotImplementedError
                self.add_fingertip_to_object_distance_constraint_to_ik(
                    ik, RobotFinger.LEFT, 0.00, 0.001
                )
            else:
                self.add_obj_env_contact_constraints_to_ik(
                    ik,
                    p_BC,
                    p_EC,
                    0.00,
                    0.001,
                    0.00,
                    0.001,
                    contact_object,
                )
        q0 = np.array([-1.57, 0.1, 0, 0, 0, 1.6, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2])
        # Solve
        if X_WB_guess is not None:
            # add cost to encourage the object to be in the same pose as the guess
            self.add_obj_X_WB_cost(
                ik,
                X_WB_guess,
            )
            q0[9:16] = math_utils.X_to_drake_pose_state(X_WB_guess)
        if franka_q_guess is not None:
            assert len(franka_q_guess) == 9
            q0[:9] = franka_q_guess
        # solve IK
        prog = ik.get_mutable_prog()
        prog.SetInitialGuess(ik.q(), franka_q_guess)
        result = mp.Solve(prog)
        if not result.is_success():
            print("IK failed")
            return None
        q_val = result.GetSolution(ik.q())
        if return_q:
            return q_val
        return (
            FrankaTableStation.get_X_WB(q_val),
            FrankaTableStation.get_quat_WB(q_val),
        )

    def get_fingertip_p_WF(self, q, finger_type):
        """
        Get the end effector pose in the world frame
        """
        assert len(q) == 16
        diagram_context = self.set_positions(q)
        plant_context = self.diagram.GetMutableSubsystemContext(
            self.plant, diagram_context
        )
        return (
            self.robot_finger_bodies[finger_type]
            .body_frame()
            .CalcPoseInWorld(plant_context)
        )

    def visualize_q(self, q):
        diagram_context = self.set_positions(q)
        self.run_simulation(0.01, diagram_context)

    @staticmethod
    def get_franka_default_arm_q():
        return np.array(
            [
                0.09162008114028396,
                -0.19826458111314524,
                -0.01990020486871322,
                -2.4732269941140346,
                -0.01307073642274261,
                2.30396583422025,
                0.8480939705504309,
            ]
        )
