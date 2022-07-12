# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from typing import List, Dict
import manifpy as manif
from dataclasses import dataclass, field

from gym_ignition.rbd.idyntree import numpy
from adherent.data_processing import utils
from gym_ignition.rbd.conversions import Rotation
from gym_ignition.rbd.conversions import Transform
from gym_ignition.rbd.conversions import Quaternion
from adherent.data_processing import motion_data
from gym_ignition.rbd.idyntree import kindyncomputations
import bipedal_locomotion_framework.bindings as blf
from adherent.data_processing.utils import Integrator
from adherent.data_processing.utils import world_gravity


@dataclass
class IKTargets:
    """Class to manipulate the targets for the IK used in the retargeting pipeline."""
    timestamps: List[float]
    root_link: str
    base_pose_targets: Dict
    link_orientation_targets: Dict

    @staticmethod
    def build(motiondata: motion_data.MotionData,
              metadata: motion_data.MocapMetadata) -> "IKTargets":
        """Build an instance of IKTargets."""

        link_orientation_targets = {}
        base_pose_targets = {}

        for link in motiondata.Links:

            link_orientation_targets[link['name']] = np.array(link['orientations'])

            if link['name'] == metadata.root_link:
                base_pose_targets['positions'] = np.array(link['positions'])
                base_pose_targets['orientations'] = np.array(link['orientations'])

        return IKTargets(timestamps=motiondata.SampleDurations,
                         base_pose_targets=base_pose_targets,
                         link_orientation_targets=link_orientation_targets,
                         root_link=metadata.root_link)

    @staticmethod
    def mirror_quat_wrt_xz_world_plane(quat) -> np.array:
        """Mirror a quaternion w.r.t. the world X-Z plane."""

        R = Rotation.from_quat(utils.to_xyzw(np.asarray(quat)))
        RPY = R.as_euler('xyz')

        mirrored_R = Rotation.from_euler('xyz', [-RPY[0], RPY[1], -RPY[2]])
        mirrored_quat = Quaternion.to_wxyz(mirrored_R.as_quat())

        return np.array(mirrored_quat)

    @staticmethod
    def mirror_pos_wrt_xz_world_plane(pos) -> np.array:
        """Mirror a position w.r.t. the world X-Z plane."""

        mirrored_pos = pos
        mirrored_pos[1] *= -1

        return np.array(mirrored_pos)

    def mirror_ik_targets(self) -> None:
        """Mirror the ik targets. The base poses are mirrored w.r.t. the world X-Z plane. The left and
        right link orientations for the limbs are switched and mirrored w.r.t. the model's sagittal plane.
        """

        # Define a mapping between the links in order to exchange left and right limbs
        link_to_link_mapping = {link:
                                    "Right" + link[4:] if "Left" in link
                                    else "Left" + link[5:] if "Right" in link
                                    else link
                                for link in self.link_orientation_targets}

        # ====================
        # BASE POSES MIRRORING
        # ====================

        # Replace original base positions with mirrored base positions
        base_positions = self.base_pose_targets['positions']
        mirrored_base_positions = [self.mirror_pos_wrt_xz_world_plane(np.asarray(base_pos)) for base_pos in base_positions]
        self.base_pose_targets['positions'] = mirrored_base_positions

        # Replace original base orientations with mirrored base orientations
        base_orientations = self.base_pose_targets['orientations']
        mirrored_base_orientations = [self.mirror_quat_wrt_xz_world_plane(np.asarray(base_quat)) for base_quat in base_orientations]
        self.base_pose_targets['orientations'] = mirrored_base_orientations

        # Store mirrored base (B) orientations w.r.t. world (W) and original world orientations w.r.t. base (used below)
        mirrored_W_Rs_B = [Rotation.from_quat(Quaternion.to_xyzw(base_quat)) for base_quat in mirrored_base_orientations]
        original_B_Rs_W = [Rotation.from_matrix(np.linalg.inv(Rotation.from_quat(Quaternion.to_xyzw(base_quat)).as_matrix()))
                           for base_quat in base_orientations]

        # ===========================
        # LINK ORIENTATIONS MIRRORING
        # ===========================

        original_orientations = self.link_orientation_targets.copy()

        for link in self.link_orientation_targets:

            # Skip the root link
            if link == self.root_link:
                continue

            # Match link with its mirror link according to the predefined mapping
            mirror_link = link_to_link_mapping[link]
            print("Assign to", link, "the references of", mirror_link)

            # Retrieve original mirror-link quaternions (in the world frame)
            W_mirror_link_quat = original_orientations[mirror_link]

            # Initialize mirrored mirror-link quaternions (in the world frame)
            W_mirror_link_mirrored_quaternions = []

            for i in range(len(W_mirror_link_quat)):

                # Compute mirror-link RPYs (in the original base frame)
                W_mirror_link_orientation = Rotation.from_quat(Quaternion.to_xyzw(np.array(W_mirror_link_quat[i])))
                B_mirror_link_orientation = Rotation.from_matrix(original_B_Rs_W[i].as_matrix().dot(W_mirror_link_orientation.as_matrix()))
                B_mirror_link_RPY = B_mirror_link_orientation.as_euler('xyz')

                # Mirror mirror-link orientation w.r.t. the model's sagittal plane (i.e. revert roll and yaw signs)
                B_mirror_link_mirrored_orientation = \
                    Rotation.from_euler('xyz', [-B_mirror_link_RPY[0], B_mirror_link_RPY[1], -B_mirror_link_RPY[2]])

                # Express the mirrored mirror-link orientation in the world frame (using the mirrored base orientation)
                W_mirror_link_mirrored_orientation = Rotation.from_matrix(mirrored_W_Rs_B[i].as_matrix().dot(B_mirror_link_mirrored_orientation.as_matrix()))

                # Retrieve quaternions and add them to the mirrored mirror-link quaternions (in the world frame)
                W_mirror_link_mirrored_quaternion = Quaternion.to_wxyz(W_mirror_link_mirrored_orientation.as_quat())
                W_mirror_link_mirrored_quaternions.append(W_mirror_link_mirrored_quaternion)

            # Assign to the link the mirrored mirror-link quaternions
            self.link_orientation_targets[link] = np.array(W_mirror_link_mirrored_quaternions)

    def enforce_horizontal_feet(self) -> None:
        """Enforce zero roll and pitch target orientation for the feet, i.e. enforce feet parallel to the ground."""

        for link in ["RightFoot", "LeftFoot"]:

            print("Enforcing horizontal", link)

            updated_orientations = []

            for i in range(len(self.link_orientation_targets[link])):

                # Retrieve original target yaw
                original_quaternions = self.link_orientation_targets[link][i]
                original_rotation = Rotation.from_quat(utils.to_xyzw(original_quaternions))
                original_yaw = original_rotation.as_euler('xyz')[2]

                # Enforce zero pitch and roll
                updated_rotation = Rotation.from_euler('xyz', [0, 0, original_yaw])
                updated_quaternions = Quaternion.to_wxyz(updated_rotation.as_quat())
                updated_orientations.append(updated_quaternions)

            self.link_orientation_targets[link] = np.array(updated_orientations)

    def enforce_straight_head(self) -> None:
        """Enforce torso roll and pitch target orientation for the head, while keeping the yaw unchanged."""

        print("Enforcing straight Head")

        updated_head_orientations = []

        for i in range(len(self.link_orientation_targets["Head"])):

            # Retrieve original head target yaw
            original_head_quaternions = self.link_orientation_targets["Head"][i]
            original_head_rotation = Rotation.from_quat(utils.to_xyzw(original_head_quaternions))
            original_head_yaw = original_head_rotation.as_euler('xyz')[2]

            # Retrieve torso target roll and pitch
            torso_quaternions = self.link_orientation_targets["T8"][i]
            torso_rotation = Rotation.from_quat(utils.to_xyzw(torso_quaternions))
            torso_euler_angles = torso_rotation.as_euler('xyz')
            torso_roll = torso_euler_angles[0]
            torso_pitch = torso_euler_angles[1]

            # Enforce torso roll and pitch target orientation for the head, while keeping the yaw unchanged
            updated_head_rotation = Rotation.from_euler('xyz', [torso_roll, torso_pitch, original_head_yaw])
            updated_head_quaternions = Quaternion.to_wxyz(updated_head_rotation.as_quat())
            updated_head_orientations.append(updated_head_quaternions)

        self.link_orientation_targets["Head"] = np.array(updated_head_orientations)

    def enforce_wider_legs(self) -> None:
        """Enforce offsets to the upper leg target orientations so to avoid feet crossing."""

        for link in ["LeftUpperLeg", "RightUpperLeg"]:

            print("Enforcing wider", link)

            updated_orientations = []

            for i in range(len(self.link_orientation_targets[link])):

                # Retrieve original target rpy
                original_quaternions = self.link_orientation_targets[link][i]
                original_rotation = Rotation.from_quat(utils.to_xyzw(original_quaternions))
                original_rpy = original_rotation.as_euler('xyz')

                # Enforce RPY offsets # TODO
                if link == "LeftUpperLeg":
                    original_rpy += [0.1, 0, 0]
                elif link == "RightUpperLeg":
                    original_rpy -= [0.1, 0, 0]
                updated_rotation = Rotation.from_euler('xyz', original_rpy)
                updated_quaternions = Quaternion.to_wxyz(updated_rotation.as_quat())
                updated_orientations.append(updated_quaternions)

            self.link_orientation_targets[link] = np.array(updated_orientations)


@dataclass
class IKElement:
    """Class for the inverse kinematics IK"""

    # Inverse kinematics
    ik_element: blf.ik.QPInverseKinematics

    # Dictionary
    temp_dict: Dict

    # Task for the pose (IK)
    pose_se3_task: blf.ik.SE3Task = field(default_factory=lambda: blf.ik.SE3Task())

    @staticmethod
    def build(motiondata: motion_data.MotionData) -> "IKElement":
        """Build an instance of IKElement"""

        # Set ik parameters
        ik_element_param_handler = blf.parameters_handler.StdParametersHandler()
        ik_element_param_handler.set_parameter_string(name="robot_velocity_variable_name", value="robotVelocity")

        # Initialize ik element
        ik_element = blf.ik.QPInverseKinematics()
        assert ik_element.initialize(handler=ik_element_param_handler)

        temp_dict = {}

        for link in motiondata.Links:
            orientation_so3_task: blf.ik.SO3Task = blf.ik.SO3Task()
            temp_dict[link['name']] = orientation_so3_task

        return IKElement(ik_element=ik_element, temp_dict=temp_dict)

    def configure_pose_task(self, kindyn: blf.floating_base_estimators.KinDynComputations, joints_list: List) -> None:
        """Configure pose task"""

        # Configure pose task
        pose_se3_param_handler = blf.parameters_handler.StdParametersHandler()
        pose_se3_param_handler.set_parameter_string(name="robot_velocity_variable_name", value="robotVelocity")
        pose_se3_param_handler.set_parameter_string(name="frame_name", value="data_Pelvis")
        pose_se3_param_handler.set_parameter_float(name="kp_linear", value=7.5)
        pose_se3_param_handler.set_parameter_float(name="kp_angular", value=10.0)
        assert self.pose_se3_task.set_kin_dyn(kindyn)
        assert self.pose_se3_task.initialize(param_handler=pose_se3_param_handler)
        pose_se3_var_handler = blf.system.VariablesHandler()
        assert pose_se3_var_handler.add_variable("robotVelocity", len(joints_list) + 6) is True
        assert self.pose_se3_task.set_variables_handler(variables_handler=pose_se3_var_handler)

        # Add pose SE3 task as hard constraint
        assert self.ik_element.add_task(task=self.pose_se3_task, taskName="pose_se3_task", priority=0)

    def configure_orientation_task(self, kindyn: blf.floating_base_estimators.KinDynComputations, joints_list: List) -> None:
        """Configure orientation SO3 task and add it as soft constraint."""

        # Configure SO3 Tasks
        for key in self.temp_dict.keys():

            # Skip the Pelvis link
            if key == "Pelvis":
                continue

            orientation_so3_param_handler = blf.parameters_handler.StdParametersHandler()
            orientation_so3_param_handler.set_parameter_string(name="robot_velocity_variable_name", value="robotVelocity")
            orientation_so3_param_handler.set_parameter_string(name="frame_name", value=f"data_{key}")
            orientation_so3_param_handler.set_parameter_float(name="kp_angular", value=10.0)
            assert self.temp_dict[key].set_kin_dyn(kindyn)
            assert self.temp_dict[key].initialize(param_handler=orientation_so3_param_handler)
            orientation_so3_var_handler = blf.system.VariablesHandler()
            assert orientation_so3_var_handler.add_variable("robotVelocity", len(joints_list) + 6) is True
            assert self.temp_dict[key].set_variables_handler(variables_handler=orientation_so3_var_handler)

            # Add orientation SO3 task as soft constraint
            assert self.ik_element.add_task(task=self.temp_dict[key], taskName=f"orientation_so3_task_{key}", priority=1, weight=[10, 10, 10])

    def finalize_ik(self, joints_list: List) -> None:
        """Once all the tasks have been added, finalize the ik."""

        # Finalize the inverse kinematics
        ik_element_var_handler = blf.system.VariablesHandler()
        assert ik_element_var_handler.add_variable("robotVelocity", len(joints_list) + 6) is True
        assert self.ik_element.finalize(handler=ik_element_var_handler)

    def set_pose_set_point(self, target_base_position: np.ndarray, target_base_quaternion: np.ndarray):
        """Set set point for the pose SE3 task"""

        original_quaternion = utils.to_xyzw(target_base_quaternion)
        I_H_F = manif.SE3(position=target_base_position, quaternion=original_quaternion)
        mixed_velocity = manif.SE3Tangent([0.0] * 6)
        assert self.pose_se3_task.set_set_point(I_H_F=I_H_F, mixed_velocity=mixed_velocity)

    def set_orientation_set_point(self, target_link_quaternion: np.ndarray, temp_link: str):
        """Set set point for the orientation SO3 task"""

        original_quaternion = utils.to_xyzw(target_link_quaternion)
        I_R_F = manif.SO3(quaternion=original_quaternion)
        angularVelocity = manif.SO3Tangent([0.0] * 3)
        assert self.temp_dict[temp_link].set_set_point(I_R_F=I_R_F, angular_velocity=angularVelocity)

    def solve_ik(self) -> np.array:
        """Solve the ik problem and return the solution (desired joint velocities)."""

        try:
            # Solve the inverse kinematics
            assert self.ik_element.advance()
        except:
            # If IK fails to find a solution, trajectory tracking is interrupted
            print("IK solution not found. Trajectory tracking interrupted.")

        # Get the inverse kinematics output
        ik_state = self.ik_element.get_output()
        assert self.ik_element.is_output_valid()

        return ik_state.joint_velocity


@dataclass
class IKFinalSol:
    """Class for the full IK solution"""

    joint_configuration_sol: np.ndarray = np.zeros(32)
    base_position_sol: np.ndarray = np.zeros(3)
    base_quaternion_sol: np.ndarray = np.zeros(4)


@dataclass
class WBGR:
    """Class implementing the Whole-Body Geometric Retargeting (WBGR)."""

    # Robot model and joints
    icub_urdf: str
    joints_list: List

    # Components of WBGR
    ik_targets: IKTargets
    ik: IKElement
    robot_to_target_base_quat: List
    joints_integrator: Integrator = None

    # Kindyn descriptor
    kindyn_des_desc: blf.floating_base_estimators.KinDynComputationsDescriptor = None

    # Desired quantities
    joints_values_des: np.array = field(default_factory=lambda: np.array([]))
    joints_velocities_des: np.array = field(default_factory=lambda: np.array([]))
    base_values_des: np.array = field(default_factory=lambda: np.array([]))

    # Measured quantities
    joints_initial_velocities: np.array = np.zeros(32)
    base_twist: np.array = np.zeros(6)
    joints_values: np.array = np.array([0.0899, 0.0233, -0.0046, -0.5656, -0.3738, -0.0236,
                              0.0899, 0.0233, -0.0046, -0.5656, -0.3738, -0.0236,
                              0.1388792845, 0.0, 0.0,
                              -0.0629, 0.4397, 0.1825, 0.5387, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0,
                              -0.0629, 0.4397, 0.1825, 0.5387, 0.0, 0.0, 0.0])

    # Other parameters
    dt: float = 0.01

    # Base parameters
    initial_rotation_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    rotation_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    initial_position = [0, 0, 0]
    last_row = [0, 0, 0, 1]
    world_H_base = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    @staticmethod
    def build(icub_urdf: str,
              motiondata: motion_data.MotionData,
              metadata: motion_data.MocapMetadata,
              controlled_joints: List,
              mirroring: bool = False,
              horizontal_feet: bool = False,
              straight_head: bool = False,
              wider_legs: bool = False,
              robot_to_target_base_quat: List = None) -> "WBGR":
        """Build an instance of WBGR."""

        # Instantiate IKTargets
        ik_targets = IKTargets.build(motiondata=motiondata, metadata=metadata)

        # Instantiate IKElement
        ik = IKElement.build(motiondata=motiondata)

        if mirroring:
            # Mirror the ik targets
            ik_targets.mirror_ik_targets()

        if horizontal_feet:
            # Enforce feet parallel to the ground
            ik_targets.enforce_horizontal_feet()

        if straight_head:
            # Enforce straight head
            ik_targets.enforce_straight_head()

        if wider_legs:
            # Enforce wider legs
            ik_targets.enforce_wider_legs()

        return WBGR(icub_urdf=icub_urdf, joints_list=controlled_joints, ik_targets=ik_targets, ik=ik, robot_to_target_base_quat=robot_to_target_base_quat)

    # =============
    # CONFIGURATION
    # =============

    def configure(self) -> None:
        """Configure the entire pipeline."""

        # Configure the kindyn descriptors
        self.configure_kindyn_descriptors()

        # Set the initial position vector
        self.initial_position = self.ik_targets.base_pose_targets['positions'][0]

        # Retrieve initial rotation matrix
        initial_r = Rotation.from_quat(self.ik_targets.base_pose_targets['orientations'][0])
        self.initial_rotation_matrix = initial_r.as_matrix()

        # Initialize the transformation matrix
        self.setupTransformationMatrix()

        # Configure the robot state
        assert self.kindyn_des_desc.kindyn.set_robot_state(self.world_H_base, self.joints_values, self.base_twist, self.joints_initial_velocities, world_gravity())

        # Configure the inverse kinematics
        self.configure_blf_ik()

        # Configure the integrator from joint velocities to joint positions
        self.configure_joints_integrator()

    def setupTransformationMatrix(self) -> None:
        """Initialize the transformation matrix"""

        M = len(self.world_H_base)
        N = len(self.initial_rotation_matrix)

        for i in range(M):
            self.world_H_base[3][i] = self.last_row[i]

        for i in range(N):
            self.world_H_base[i][3] = self.initial_position[i]

        for i in range(N):
            for j in range(N):
                self.world_H_base[i][j] = self.initial_rotation_matrix[i][j]

    def configure_kindyn_descriptors(self) -> None:
        """Setup the kindyn descriptor for desired values."""

        # create KinDynComputationsDescriptor
        kindyn_des_handler = blf.parameters_handler.StdParametersHandler()
        kindyn_des_handler.set_parameter_string("model_file_name", self.icub_urdf)
        kindyn_des_handler.set_parameter_vector_string("joints_list", self.joints_list)
        self.kindyn_des_desc = blf.floating_base_estimators.construct_kindyncomputations_descriptor(kindyn_des_handler)
        assert self.kindyn_des_desc.is_valid()
        self.kindyn_des_desc.kindyn.set_floating_base("root_link")

    def configure_blf_ik(self) -> None:
        """Setup the inverse kinematics by adding tasks and setting fixed set points."""

        # Configure pose task
        self.ik.configure_pose_task(kindyn=self.kindyn_des_desc.kindyn, joints_list=self.joints_list)

        # Configure orientation task
        self.ik.configure_orientation_task(kindyn=self.kindyn_des_desc.kindyn, joints_list=self.joints_list)

        # Finalize the ik
        self.ik.finalize_ik(joints_list=self.joints_list)

    def configure_joints_integrator(self) -> None:
        """Initialize the integrator for the joint references."""

        self.joints_integrator = Integrator.build(joints_initial_position=self.joints_values, dt=self.dt)

    # ========
    # RETARGET
    # ========

    def retarget(self) -> (List, List):
        """Apply Whole-Body Geometric Retargeting (WBGR)."""

        ik_solutions = []
        timestamps = []

        for i in range(len(self.ik_targets.timestamps)):

            print(i, "/", len(self.ik_targets.timestamps))

            timestamps.append(self.ik_targets.timestamps[i])

            # ==============
            # UPDATE TARGETS
            # ==============

            # Base pose target
            self.ik.set_pose_set_point(target_base_position=self.ik_targets.base_pose_targets['positions'][i], target_base_quaternion=self.ik_targets.base_pose_targets['orientations'][i])

            # Link orientation targets
            for link, orientations in self.ik_targets.link_orientation_targets.items():

                # Skip the root link
                if link == self.ik_targets.root_link:
                    continue

                self.ik.set_orientation_set_point(target_link_quaternion=orientations[i, :], temp_link=link)

            # ========
            # SOLVE IK
            # ========

            # Retrieve joint velocities
            self.joints_velocities_des = self.ik.solve_ik()
            joints_velocities_copy = np.copy(self.joints_velocities_des)

            # Retrieve joint positions
            self.joints_integrator.advance(joints_velocity=self.joints_velocities_des)
            self.joints_values_des = self.joints_integrator.get_joints_position()
            joints_values_copy = np.copy(self.joints_values_des)

            # Position of the base of the robot (fixed)
            base_fixed_position = [0, 0, 0.53]

            # Retrieve base orientation
            human_base_orientation = self.ik_targets.base_pose_targets['orientations'][i]

            # Retrieve rotation matrix
            rotated_quaternion = utils.to_xyzw(human_base_orientation)
            temp_human_base_orientation = Rotation.from_quat(rotated_quaternion)
            self.rotation_matrix = temp_human_base_orientation.as_matrix()

            # Update the world_H_base
            M = len(self.world_H_base)
            N = len(self.rotation_matrix)

            for index in range(M):
                self.world_H_base[3][index] = self.last_row[index]

            for index in range(N):
                self.world_H_base[index][3] = base_fixed_position[index]

            for index in range(N):
                for j in range(N):
                    self.world_H_base[index][j] = self.rotation_matrix[index][j]

            # Update ik solution
            ik_solutions.append(IKFinalSol(joint_configuration_sol=joints_values_copy, base_position_sol=base_fixed_position, base_quaternion_sol=human_base_orientation))

            # Update the robot state
            assert self.kindyn_des_desc.kindyn.set_robot_state(self.world_H_base, joints_values_copy, self.base_twist, joints_velocities_copy, world_gravity())

        return timestamps, ik_solutions


@dataclass
class KinematicComputations:
    """Class for the kinematic computations exploited by the Kinematically-Feasible Whole-Body
    Geometric Retargeting (KFWBGR).
    """

    kindyn: kindyncomputations.KinDynComputations
    local_foot_vertices_pos: List
    feet_frames: Dict

    @staticmethod
    def build(kindyn: kindyncomputations.KinDynComputations,
              local_foot_vertices_pos: List, feet_frames: Dict) -> "KinematicComputations":
        """Build an instance of KinematicComputations."""

        return KinematicComputations(kindyn=kindyn, local_foot_vertices_pos=local_foot_vertices_pos, feet_frames=feet_frames)

    def compute_W_vertices_pos(self) -> List:
        """Compute the feet vertices positions in the world (W) frame."""

        # Retrieve front-left (FL), front-right (FR), back-left (BL) and back-right (BR) vertices in the foot frame
        FL_vertex_pos = self.local_foot_vertices_pos[0]
        FR_vertex_pos = self.local_foot_vertices_pos[1]
        BL_vertex_pos = self.local_foot_vertices_pos[2]
        BR_vertex_pos = self.local_foot_vertices_pos[3]

        # Compute right foot (RF) transform w.r.t. the world (W) frame
        world_H_base = self.kindyn.get_world_base_transform()
        base_H_r_foot = self.kindyn.get_relative_transform(ref_frame_name="root_link", frame_name=self.feet_frames["right_foot"])
        W_H_RF = world_H_base.dot(base_H_r_foot)

        # Get the right-foot vertices positions in the world frame
        W_RFL_vertex_pos_hom = W_H_RF @ np.concatenate((FL_vertex_pos, [1]))
        W_RFR_vertex_pos_hom = W_H_RF @ np.concatenate((FR_vertex_pos, [1]))
        W_RBL_vertex_pos_hom = W_H_RF @ np.concatenate((BL_vertex_pos, [1]))
        W_RBR_vertex_pos_hom = W_H_RF @ np.concatenate((BR_vertex_pos, [1]))

        # Convert homogeneous to cartesian coordinates
        W_RFL_vertex_pos = W_RFL_vertex_pos_hom[0:3]
        W_RFR_vertex_pos = W_RFR_vertex_pos_hom[0:3]
        W_RBL_vertex_pos = W_RBL_vertex_pos_hom[0:3]
        W_RBR_vertex_pos = W_RBR_vertex_pos_hom[0:3]

        # Compute left foot (LF) transform w.r.t. the world (W) frame
        world_H_base = self.kindyn.get_world_base_transform()
        base_H_l_foot = self.kindyn.get_relative_transform(ref_frame_name="root_link", frame_name=self.feet_frames["left_foot"])
        W_H_LF = world_H_base.dot(base_H_l_foot)

        # Get the left-foot vertices positions wrt the world frame
        W_LFL_vertex_pos_hom = W_H_LF @ np.concatenate((FL_vertex_pos, [1]))
        W_LFR_vertex_pos_hom = W_H_LF @ np.concatenate((FR_vertex_pos, [1]))
        W_LBL_vertex_pos_hom = W_H_LF @ np.concatenate((BL_vertex_pos, [1]))
        W_LBR_vertex_pos_hom = W_H_LF @ np.concatenate((BR_vertex_pos, [1]))

        # Convert homogeneous to cartesian coordinates
        W_LFL_vertex_pos = W_LFL_vertex_pos_hom[0:3]
        W_LFR_vertex_pos = W_LFR_vertex_pos_hom[0:3]
        W_LBL_vertex_pos = W_LBL_vertex_pos_hom[0:3]
        W_LBR_vertex_pos = W_LBR_vertex_pos_hom[0:3]

        # Store the positions of both right-foot and left-foot vertices in the world frame
        W_vertices_positions = [W_RFL_vertex_pos, W_RFR_vertex_pos, W_RBL_vertex_pos, W_RBR_vertex_pos,
                                W_LFL_vertex_pos, W_LFR_vertex_pos, W_LBL_vertex_pos, W_LBR_vertex_pos]

        return W_vertices_positions

    def reset_robot_configuration(self, joint_positions: List, base_position: List, base_quaternion: List) -> None:
        """Reset the robot configuration."""

        world_H_base = numpy.FromNumPy.to_idyntree_transform(
            position=np.array(base_position),
            quaternion=np.array(base_quaternion)).asHomogeneousTransform().toNumPy()

        self.kindyn.set_robot_state(s=joint_positions, ds=np.zeros(len(joint_positions)), world_H_base=world_H_base)

    def compute_support_vertex_pos(self, support_foot: str, support_vertex: int) -> List:
        """Compute the support vertex position in the world (W) frame."""

        # Compute the transform of the support foot (SF) wrt the world (W) frame
        world_H_base = self.kindyn.get_world_base_transform()
        base_H_SF = self.kindyn.get_relative_transform(ref_frame_name="root_link", frame_name=support_foot)
        W_H_SF = world_H_base.dot(base_H_SF)

        # Get the support vertex position wrt the world frame
        F_support_vertex_pos = self.local_foot_vertices_pos[support_vertex]
        F_support_vertex_pos_hom = np.concatenate((F_support_vertex_pos, [1]))
        W_support_vertex_pos_hom = W_H_SF @ F_support_vertex_pos_hom
        W_support_vertex_pos = W_support_vertex_pos_hom[0:3]

        return W_support_vertex_pos

    def compute_base_position_by_leg_odometry(self, support_vertex_pos: List, support_foot: str,
                                              support_vertex_offset: List) -> List:
        """Compute kinematically-feasible base position using leg odometry."""

        # Get the base (B) position in the world (W) frame
        W_pos_B = self.kindyn.get_world_base_transform()[0:3, -1]

        # Get the support vertex position in the world (W) frame
        W_support_vertex_pos = support_vertex_pos

        # Get the support vertex orientation in the world (W) frame, defined as the support foot (SF) orientation
        world_H_base = self.kindyn.get_world_base_transform()
        base_H_SF = self.kindyn.get_relative_transform(ref_frame_name="root_link", frame_name=support_foot)
        W_H_SF = world_H_base.dot(base_H_SF)
        W_support_vertex_quat = Quaternion.from_matrix(W_H_SF[0:3, 0:3])

        # Compute the transform of the support vertex (SV) in the world (W) frame
        W_H_SV = Transform.from_position_and_quaternion(position=np.asarray(W_support_vertex_pos),
                                                        quaternion=np.asarray(W_support_vertex_quat))

        # Express the base (B) position in the support vertex (SV) reference frame
        SV_H_W = np.linalg.inv(W_H_SV)
        W_pos_B_hom = np.concatenate((W_pos_B, [1]))
        SV_pos_B_hom = SV_H_W @ W_pos_B_hom

        # Express the base (B) position in a reference frame oriented as the world but positioned in the support vertex (SV)
        mixed_H_SV = Transform.from_position_and_quaternion(position=np.asarray([0, 0, 0]),
                                                            quaternion=np.asarray(W_support_vertex_quat))
        mixed_pos_B_hom = mixed_H_SV @ SV_pos_B_hom

        # Convert homogeneous to cartesian coordinates
        mixed_pos_B = mixed_pos_B_hom[0:3]

        # Compute the kinematically-feasible base position, i.e. the base position such that the support
        # vertex remains fixed while the robot configuration changes
        b_pos = mixed_pos_B + support_vertex_offset

        return b_pos


@dataclass
class KFWBGR(WBGR):
    """Class implementing the Kinematically-Feasible Whole-Body Geometric Retargeting (KFWBGR)."""

    kinematic_computations: KinematicComputations = field(default_factory= lambda: KinematicComputations())

    @staticmethod
    def build(icub_urdf: str,
              motiondata: motion_data.MotionData,
              metadata: motion_data.MocapMetadata,
              controlled_joints: List,
              mirroring: bool = False,
              horizontal_feet: bool = False,
              straight_head: bool = False,
              wider_legs: bool = False,
              robot_to_target_base_quat: List = None,
              kindyn: kindyncomputations.KinDynComputations = None,
              local_foot_vertices_pos: List = None,
              feet_frames: Dict = None) -> "KFWBGR":
        """Build an instance of KFWBGR."""

        # Instantiate IKTargets
        ik_targets = IKTargets.build(motiondata=motiondata, metadata=metadata)

        # Instantiate IKElement
        ik = IKElement.build(motiondata=motiondata)

        if mirroring:
            # Mirror the ik targets
            ik_targets.mirror_ik_targets()

        if horizontal_feet:
            # Enforce feet parallel to the ground
            ik_targets.enforce_horizontal_feet()

        if straight_head:
            # Enforce straight head
            ik_targets.enforce_straight_head()

        if wider_legs:
            # Enforce wider legs
            ik_targets.enforce_wider_legs()

        kinematic_computations = KinematicComputations.build(kindyn=kindyn, local_foot_vertices_pos=local_foot_vertices_pos, feet_frames=feet_frames)

        return KFWBGR(icub_urdf=icub_urdf, joints_list=controlled_joints, ik_targets=ik_targets, ik=ik, robot_to_target_base_quat=robot_to_target_base_quat,
                      kinematic_computations=kinematic_computations)

    def KF_retarget(self) -> (List, List):
        """Apply Kinematically-Feasible Whole-Body Geometric Retargeting (KFWBGR)."""

        # WBGR
        timestamps, ik_solutions = self.retarget()

        # Compute kinematically-feasible linear base motion
        kinematically_feasible_base_position = self.compute_kinematically_feasible_base_motion(ik_solutions)

        # Override base position in the ik solutions
        for i in range(len(kinematically_feasible_base_position)):
            ik_solutions[i+1].base_position_sol = kinematically_feasible_base_position[i]

        return timestamps, ik_solutions

    def compute_kinematically_feasible_base_motion(self, ik_solutions: List) -> List:
        """Compute a kinematically-feasible base linear motion."""

        # Initialize output list
        kinematically_feasible_base_positions = []

        # Set initial robot configuration
        initial_joint_configuration = ik_solutions[0].joint_configuration_sol
        initial_base_position = self.kinematic_computations.kindyn.get_world_base_transform()[0:3, -1]
        initial_base_quaternion = ik_solutions[0].base_quaternion_sol
        self.kinematic_computations.reset_robot_configuration(joint_positions=initial_joint_configuration,
                                                              base_position=initial_base_position,
                                                              base_quaternion=initial_base_quaternion)

        # Associate indexes to feet vertices names, from Right-foot Front Left (RFL) to Left-foot Back Right (LBR)
        vertex_indexes_to_names = {0: "RFL", 1: "RFR", 2: "RBL", 3: "RBR",
                                   4: "LFL", 5: "LFR", 6: "LBL", 7: "LBR"}

        # Define the initial support vertex index and the initial support foot
        support_vertex_prev = 0 # i.e. right-foot front-left vertex (RFL)
        support_foot = self.kinematic_computations.feet_frames["right_foot"]

        # Compute the initial support vertex position in the world frame and its ground projection
        support_vertex_pos = self.kinematic_computations.compute_support_vertex_pos(
            support_foot=support_foot, support_vertex=support_vertex_prev)
        support_vertex_offset = [support_vertex_pos[0], support_vertex_pos[1], 0]

        for i in range(1, len(ik_solutions)):

            # ===========================================
            # UPDATE JOINT POSITIONS AND BASE ORIENTATION
            # ===========================================

            joint_positions = ik_solutions[i].joint_configuration_sol
            base_quaternion = ik_solutions[i].base_quaternion_sol
            previous_base_position = self.kinematic_computations.kindyn.get_world_base_transform()[0:3, -1]
            self.kinematic_computations.reset_robot_configuration(joint_positions=joint_positions,
                                                                  base_position=previous_base_position,
                                                                  base_quaternion=base_quaternion)

            # ======================================
            # UPDATE SUPPORT VERTEX AND SUPPORT FOOT
            # ======================================

            # Retrieve the vertices positions in the world (W) frame
            W_vertices_positions = self.kinematic_computations.compute_W_vertices_pos()

            # Compute the current support vertex as the lowest among the feet vertices
            vertices_heights = [W_vertex[2] for W_vertex in W_vertices_positions]
            support_vertex = np.argmin(vertices_heights)

            if support_vertex == support_vertex_prev:

                # Update the support vertex position only
                support_vertex_pos = W_vertices_positions[support_vertex]

            else:

                # Update the support foot
                if vertex_indexes_to_names[support_vertex][0] == "R":
                    support_foot = self.kinematic_computations.feet_frames["right_foot"]
                else:
                    support_foot = self.kinematic_computations.feet_frames["left_foot"]

                # Debug
                print("Change of support vertex: from", vertex_indexes_to_names[support_vertex_prev],
                      "to", vertex_indexes_to_names[support_vertex])
                print("New support foot:", support_foot)

                # Update the support vertex position and its ground projection
                support_vertex_pos = W_vertices_positions[support_vertex]
                support_vertex_offset = [support_vertex_pos[0], support_vertex_pos[1], 0]

                support_vertex_prev = support_vertex

            # =======================
            # RECOMPUTE BASE POSITION
            # =======================

            # Compute kinematically-feasible base position by leg odometry
            kinematically_feasible_base_position = self.kinematic_computations.compute_base_position_by_leg_odometry(
                support_vertex_pos=support_vertex_pos,
                support_foot=support_foot,
                support_vertex_offset=support_vertex_offset)
            kinematically_feasible_base_positions.append(kinematically_feasible_base_position)

            # Update the robot configuration with the kinematically-feasible base position
            self.kinematic_computations.reset_robot_configuration(joint_positions=joint_positions,
                                                                  base_position=kinematically_feasible_base_position,
                                                                  base_quaternion=base_quaternion)

        return kinematically_feasible_base_positions