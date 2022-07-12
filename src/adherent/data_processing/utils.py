# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import json
from scenario import core
from scenario import gazebo as scenario
import math
import time
import yarp
import numpy as np
from typing import List, Dict
from dataclasses import dataclass
import bipedal_locomotion_framework.bindings as blf

import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt

# =====================
# MODEL INSERTION UTILS
# =====================

class iCub(core.Model):
    """Helper class to simplify model insertion."""

    def __init__(self,
                 world: scenario.World,
                 urdf: str,
                 position: List[float] = (0., 0, 0),
                 orientation: List[float] = (1., 0, 0, 0)):

        # Insert the model in the world
        name = "iCub"
        pose = core.Pose(position, orientation)
        world.insert_model(urdf, pose, name)

        # Get and store the model from the world
        self.model = world.get_model(model_name=name)

    def __getattr__(self, name):
        return getattr(self.model, name)

# =================
# RETARGETING UTILS
# =================

def define_robot_to_target_base_quat(robot: str) -> List:
    """Define the robot-specific quaternions from the robot base frame to the target base frame."""

    if robot == "iCubV2_5":
        # For iCubV2_5, the robot base frame is rotated of -180 degs on z w.r.t. the target base frame
        robot_to_target_base_quat = [0, 0, 0, -1.0]

    elif robot == "iCubV3":
        # For iCubV3, the robot base frame is the same as the target base frame
        robot_to_target_base_quat = [0, 0, 0, 0.0]

    else:
        raise Exception("Quaternions from the robot to the target base frame only defined for iCubV2_5 and iCubV3.")

    return robot_to_target_base_quat

def define_feet_frames(robot: str) -> Dict:
    """Define the robot-specific feet frames."""

    if robot == "iCubV2_5":
        right_foot = "r_foot"
        left_foot = "l_foot"

    elif robot == "iCubV3":
        right_foot = "r_foot_rear"
        left_foot = "l_foot_rear"

    else:
        raise Exception("Feet frames only defined for iCubV2_5 and iCubV3.")

    return {"right_foot": right_foot, "left_foot": left_foot}

def define_foot_vertices(robot: str) -> List:
    """Define the robot-specific positions of the feet vertices in the foot frame."""

    if robot == "iCubV2_5":

        # For iCubV2_5, the feet vertices are not symmetrically placed wrt the foot frame origin.
        # The foot frame has z pointing down, x pointing forward and y pointing right.

        # Origin of the box which represents the foot (in the foot frame)
        box_origin = [0.03, 0.005, 0.014]

        # Size of the box which represents the foot
        box_size = [0.16, 0.072, 0.001]

        # Define front-left (FL), front-right (FR), back-left (BL) and back-right (BR) vertices in the foot frame
        FL_vertex_pos = [box_origin[0] + box_size[0] / 2, box_origin[1] - box_size[1] / 2, box_origin[2]]
        FR_vertex_pos = [box_origin[0] + box_size[0] / 2, box_origin[1] + box_size[1] / 2, box_origin[2]]
        BL_vertex_pos = [box_origin[0] - box_size[0] / 2, box_origin[1] - box_size[1] / 2, box_origin[2]]
        BR_vertex_pos = [box_origin[0] - box_size[0] / 2, box_origin[1] + box_size[1] / 2, box_origin[2]]

    elif robot == "iCubV3":

        # TODO: for iCubV3, the considered foot frame is the foot rear
        # For iCubV3, the feet vertices are not symmetrically placed wrt the foot rear frame origin.
        # The foot rear frame has z pointing up, x pointing forward and y pointing left.

        # Origin of the box which represents the foot rear (in the foot frame)
        box_origin = [0.0, 0.0, 0.003]

        # Size of the box which represents the foot rear
        box_size = [0.117, 0.1, 0.006]

        # Distance between the foot rear and the foot front boxes
        boxes_distance = 0.002

        # Define front-left (FL), front-right (FR), back-left (BL) and back-right (BR) vertices in the foot frame
        FL_vertex_pos = [box_origin[0] + box_size[0] / 2 + boxes_distance + box_size[0],
                         box_origin[1] + box_size[1] / 2, box_origin[2]]
        FR_vertex_pos = [box_origin[0] + box_size[0] / 2 + boxes_distance + box_size[0],
                         box_origin[1] - box_size[1] / 2, box_origin[2]]
        BL_vertex_pos = [box_origin[0] - box_size[0] / 2, box_origin[1] + box_size[1] / 2, box_origin[2]]
        BR_vertex_pos = [box_origin[0] - box_size[0] / 2, box_origin[1] - box_size[1] / 2, box_origin[2]]

    else:
        raise Exception("Feet vertices positions only defined for iCubV2_5 and iCubV3.")

    # Vertices positions in the foot (F) frame
    F_vertices_pos = [FL_vertex_pos, FR_vertex_pos, BL_vertex_pos, BR_vertex_pos]

    return F_vertices_pos

def quaternion_multiply(quat1: List, quat2: List) -> np.array:
    """Auxiliary function for quaternion multiplication."""

    w1, x1, y1, z1 = quat1
    w2, x2, y2, z2 = quat2

    res = np.array([-x1 * x2 - y1 * y2 - z1 * z2 + w1 * w2,
                     x1 * w2 + y1 * z2 - z1 * y2 + w1 * x2,
                     -x1 * z2 + y1 * w2 + z1 * x2 + w1 * y2,
                     x1 * y2 - y1 * x2 + z1 * w2 + w1 * z2])

    return res

def to_xyzw(wxyz: List) -> List:
    """Auxiliary function to convert quaternions from wxyz to xyzw format."""

    return wxyz[[1, 2, 3, 0]]

def to_wxyz(xyzw: List) -> List:
    """Auxiliary function to convert quaternions from xyzw to wxyz format."""

    temp_list = [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]
    return temp_list

def store_retargeted_mocap_as_json(timestamps: List, ik_solutions: List, outfile_name: str) -> None:
    """Auxiliary function to store the retargeted motion."""

    ik_solutions_json = []

    for i in range(1, len(ik_solutions)):

        ik_solution = ik_solutions[i]

        ik_solution_json = {"joint_positions": ik_solution.joint_configuration_sol.tolist(),
                            "base_position": ik_solution.base_position_sol.tolist(),
                            "base_quaternion": ik_solution.base_quaternion_sol.tolist(),
                            "timestamp": timestamps[i]}

        ik_solutions_json.append(ik_solution_json)

    with open(outfile_name, "w") as outfile:
        json.dump(ik_solutions_json, outfile)

def load_retargeted_mocap_from_json(input_file_name: str, initial_frame: int = 0, final_frame: int = -1) -> (List, List):
    """Auxiliary function to load the retargeted mocap data."""

    # Load ik solutions
    with open(input_file_name, 'r') as openfile:
        ik_solutions = json.load(openfile)

    # If a final frame has been passed, extract relevant ik solutions
    if initial_frame != -1:
        ik_solutions = ik_solutions[initial_frame:final_frame]

    # Extract timestamps
    timestamps = [ik_solution["timestamp"] for ik_solution in ik_solutions]

    return timestamps, ik_solutions

# =========================
# FEATURES EXTRACTION UTILS
# =========================

def define_frontal_base_direction(robot: str) -> List:
    """Define the robot-specific frontal base direction in the base frame."""

    if robot == "iCubV2_5":
        # For iCubV2_5, the reversed x axis of the base frame is pointing forward
        frontal_base_direction = [-1, 0, 0]

    elif robot == "iCubV3":
        # For iCubV3, the x axis is pointing forward
        frontal_base_direction = [1, 0, 0]

    else:
        raise Exception("Frontal base direction only defined for iCubV2_5 and iCubV3.")

    return frontal_base_direction

def rotateMatrix(mat: List) -> List:
    """Function to rotate a matrix by 180 degrees"""
    N = len(mat)

    for i in range(N // 2):
        for j in range(N):
            temp = mat[i][j]
            mat[i][j] = mat[N - i - 1][N - j - 1]
            mat[N - i - 1][N - j - 1] = temp


def define_frontal_chest_direction(robot: str) -> List:
    """Define the robot-specific frontal chest direction in the chest frame."""

    if robot == "iCubV2_5":
        # For iCubV2_5, the z axis of the chest frame is pointing forward
        frontal_chest_direction = [0, 0, 1]

    elif robot == "iCubV3":
        # For iCubV3, the x axis of the chest frame is pointing forward
        frontal_chest_direction = [1, 0, 0]

    else:
        raise Exception("Frontal chest direction only defined for iCubV2_5 and iCubV3.")

    return frontal_chest_direction

def rotation_2D(angle: float) -> np.array:
    """Auxiliary function for a 2-dimensional rotation matrix."""

    return np.array([[math.cos(angle), -math.sin(angle)],
                     [math.sin(angle), math.cos(angle)]])

# ===================
# VISUALIZATION UTILS
# ===================

def visualize_retargeted_motion(timestamps: List,
                                ik_solutions: List,
                                icub: iCub,
                                controlled_joints: List,
                                gazebo: scenario.GazeboSimulator) -> None:
    """Auxiliary function to visualize retargeted motion."""

    timestamp_prev = -1

    for i in range(1, len(ik_solutions)):

        ik_solution = ik_solutions[i]

        # Retrieve the base pose and the joint positions
        if type(ik_solution) == dict:
            joint_positions = ik_solution["joint_positions"]
            base_position = ik_solution["base_position"]
            base_quaternion = ik_solution["base_quaternion"]
        else:
            joint_positions = ik_solution.joint_configuration_sol
            base_position = ik_solution.base_position_sol
            base_quaternion = ik_solution.base_quaternion_sol

        # Reset the base pose and the joint positions
        icub.to_gazebo().reset_base_pose(base_position, base_quaternion)
        icub.to_gazebo().reset_joint_positions(joint_positions, controlled_joints)
        gazebo.run(paused=True)

        # Visualize the retargeted motion at the time rate of the collected data
        timestamp = timestamps[i]
        if timestamp_prev == -1:
            dt = 1 / 100
        else:
            dt = timestamp - timestamp_prev
        time.sleep(dt)
        timestamp_prev = timestamp


    print("Visualization ended")
    time.sleep(1)

def visualize_global_features(global_window_features,
                              ik_solutions: List,
                              icub: iCub,
                              controlled_joints: List,
                              gazebo: scenario.GazeboSimulator,
                              plot_facing_directions: bool = True,
                              plot_base_velocities: bool = False) -> None:
    """Visualize the retargeted frames along with the associated global features."""

    window_length_frames = global_window_features.window_length_frames
    window_step = global_window_features.window_step
    window_indexes = global_window_features.window_indexes
    initial_frame = window_length_frames
    final_frame = round(len(ik_solutions)/2) - window_length_frames - window_step - 1

    plt.ion()

    for i in range(initial_frame, final_frame):

        # Debug
        print(i - initial_frame, "/", final_frame - initial_frame)

        # The ik solutions are stored at double frequency w.r.t. the extracted features
        ik_solution = ik_solutions[2 * i]

        # Retrieve the base pose and the joint positions
        joint_positions = np.asarray(ik_solution["joint_positions"])
        base_position = np.asarray(ik_solution["base_position"])
        base_quaternion = np.asarray(ik_solution["base_quaternion"])

        # Reset the base pose and the joint positions
        icub.to_gazebo().reset_base_pose(base_position, base_quaternion)
        icub.to_gazebo().reset_joint_positions(joint_positions, controlled_joints)
        gazebo.run(paused=True)

        # Retrieve global features
        base_positions = global_window_features.base_positions[i - window_length_frames]
        facing_directions = global_window_features.facing_directions[i - window_length_frames]
        base_velocities = global_window_features.base_velocities[i - window_length_frames]

        # =================
        # FACING DIRECTIONS
        # =================

        if plot_facing_directions:

            # Figure 1 for the facing directions
            plt.figure(1)
            plt.clf()

            for j in range(len(base_positions)):

                # Plot base positions
                base_position = base_positions[j]
                if window_indexes[j] == 0:
                    # Current base position in red
                    plt.scatter(base_position[1], -base_position[0], c='r', label="Current base position")
                else:
                    # Other base positions in black
                    plt.scatter(base_position[1], -base_position[0], c='k')

                # Plot facing directions
                facing_direction = facing_directions[j] / 10  # scaled for visualization purposes
                if window_indexes[j] == 0:
                    # Current facing direction in blue
                    plt.plot([base_position[1], base_position[1] + 2 * facing_direction[1]],
                             [-base_position[0], -base_position[0] - 2 * facing_direction[0]], 'b',
                             label="Current facing direction")
                else:
                    # Other facing directions in green
                    plt.plot([base_position[1], base_position[1] + facing_direction[1]],
                             [-base_position[0], -base_position[0] - facing_direction[0]], 'g')

            # Configuration
            plt.axis('scaled')
            plt.xlim([-1.75, 1.75])
            plt.ylim([-1.75, 1.75])
            plt.title("Facing directions (global view)")
            plt.legend()

        # ===============
        # BASE VELOCITIES
        # ===============

        if plot_base_velocities:

            # Figure 2 for the base velocities
            plt.figure(2)
            plt.clf()

            for j in range(len(base_positions)):

                # Plot base positions
                base_position = base_positions[j]
                if window_indexes[j] == 0:
                    # Current base position in red
                    plt.scatter(base_position[1], -base_position[0], c='r', label="Current base position")
                else:
                    # Other base positions in black
                    plt.scatter(base_position[1], -base_position[0], c='k')

                # Plot base velocities
                base_velocity = base_velocities[j] / 10  # scaled for visualization purposes
                if window_indexes[j] == 0:
                    # Current base velocity in magenta
                    plt.plot([base_position[1], base_position[1] + base_velocity[1]],
                             [-base_position[0], -base_position[0] - base_velocity[0]], 'm',
                             label="Current base velocity")
                else:
                    # Other base velocities in gray
                    plt.plot([base_position[1], base_position[1] + base_velocity[1]],
                             [-base_position[0], -base_position[0] - base_velocity[0]], 'gray')

            # Configuration
            plt.axis('scaled')
            plt.xlim([-1.75, 1.75])
            plt.ylim([-1.75, 1.75])
            plt.title("Base velocities (global view)")
            plt.legend()

        # Plot
        plt.show()
        plt.pause(0.0001)

def visualize_local_features(local_window_features,
                             ik_solutions: List,
                             icub: iCub,
                             controlled_joints: List,
                             gazebo: scenario.GazeboSimulator,
                             plot_facing_directions: bool = True,
                             plot_base_velocities: bool = False) -> None:
    """Visualize the retargeted frames along with the associated local features."""

    window_length_frames = local_window_features.window_length_frames
    window_step = local_window_features.window_step
    window_indexes = local_window_features.window_indexes
    initial_frame = window_length_frames
    final_frame = round(len(ik_solutions)/2) - window_length_frames - window_step - 1

    plt.ion()

    for i in range(initial_frame, final_frame):

        # Debug
        print(i - initial_frame, "/", final_frame - initial_frame)

        # The ik solutions are stored at double frequency w.r.t. the extracted features
        ik_solution = ik_solutions[2 * i]

        # Retrieve the base pose and the joint positions
        joint_positions = np.asarray(ik_solution["joint_positions"])
        base_position = np.asarray(ik_solution["base_position"])
        base_quaternion = np.asarray(ik_solution["base_quaternion"])

        # Reset the base pose and the joint positions
        icub.to_gazebo().reset_base_pose(base_position, base_quaternion)
        icub.to_gazebo().reset_joint_positions(joint_positions, controlled_joints)
        gazebo.run(paused=True)

        # Retrieve local features
        base_positions = local_window_features.base_positions[i - window_length_frames]
        facing_directions = local_window_features.facing_directions[i - window_length_frames]
        base_velocities = local_window_features.base_velocities[i - window_length_frames]

        # =================
        # FACING DIRECTIONS
        # =================

        if plot_facing_directions:

            # Figure 1 for the facing directions
            plt.figure(1)
            plt.clf()

            for j in range(len(base_positions)):

                # Plot base positions
                base_position = base_positions[j]
                if window_indexes[j] == 0:
                    # Current base position in red
                    plt.scatter(-base_position[1], base_position[0], c='r', label="Current base position")
                else:
                    # Other base positions in black
                    plt.scatter(-base_position[1], base_position[0], c='k')

                # Plot facing directions
                facing_direction = facing_directions[j] / 10  # scaled for visualization purposes
                if window_indexes[j] == 0:
                    # Current facing direction in blue
                    plt.plot([-base_position[1], -base_position[1] - 2 * facing_direction[1]],
                             [base_position[0], base_position[0] + 2 * facing_direction[0]], 'b',
                             label="Current facing direction")
                else:
                    # Other facing directions in green
                    plt.plot([-base_position[1], -base_position[1] - facing_direction[1]],
                             [base_position[0], base_position[0] + facing_direction[0]], 'g')

            # Configuration
            plt.axis('scaled')
            plt.xlim([-0.75, 0.75])
            plt.ylim([-0.75, 0.75])
            plt.title("Facing directions (local view)")
            plt.legend()

        # ===============
        # BASE VELOCITIES
        # ===============

        if plot_base_velocities:

            # Figure 2 for the base velocities
            plt.figure(2)
            plt.clf()

            for j in range(len(base_positions)):

                # Plot base positions
                base_position = base_positions[j]
                if window_indexes[j] == 0:
                    # Current base position in red
                    plt.scatter(-base_position[1], base_position[0], c='r', label="Current base position")
                else:
                    # Other base positions in black
                    plt.scatter(-base_position[1], base_position[0], c='k')

                # Plot base velocities
                base_velocity = base_velocities[j] / 10  # scaled for visualization purposes
                if window_indexes[j] == 0:
                    # Current base velocity in magenta
                    plt.plot([-base_position[1], -base_position[1] - base_velocity[1]],
                             [base_position[0], base_position[0] + base_velocity[0]], 'm',
                             label="Current base velocity")
                else:
                    # Other base velocities in gray
                    plt.plot([-base_position[1], -base_position[1] - base_velocity[1]],
                             [base_position[0], base_position[0] + base_velocity[0]], 'gray')

            # Configuration
            plt.axis('scaled')
            plt.xlim([-0.75, 0.75])
            plt.ylim([-0.75, 0.75])
            plt.title("Base velocities (local view)")
            plt.legend()

        # Plot
        plt.show()
        plt.pause(0.0001)

@dataclass
class Integrator:
    """Auxiliary class implementing an Euler integrator from joint velocities to joint positions."""

    joints_position: np.array

    # Integration step
    dt: float

    @staticmethod
    def build(joints_initial_position: np.array, dt: float) -> "Integrator":
        """Build an instance of Integrator."""

        return Integrator(joints_position=joints_initial_position, dt=dt)

    def advance(self, joints_velocity: np.array) -> None:
        """Euler integration step."""

        self.joints_position += self.dt * joints_velocity

    def get_joints_position(self) -> np.ndarray:
        """Getter of the joint position."""

        return self.joints_position


def synchronize(curr_dt: float, dt: float) -> float:
    """Auxiliary function for synchronization."""

    if curr_dt+dt - yarp.now() > 0:

        # Wait the proper amount of time to be synchronized at intervals of dt
        time.sleep(curr_dt+dt - yarp.now())

    else:

        # Debug to check whether the synchronization takes place or not
        print("no synch!")

    return curr_dt+dt

def rad2deg(rad: float) -> float:
    """Auxiliary function for radians to degrees conversion."""

    return rad / math.pi * 180

def world_gravity() -> List:
    """Auxiliary function for the gravitational constant."""

    return [0.0, 0.0, -blf.math.StandardAccelerationOfGravitation]

def define_foot_name_to_index_mapping(robot: str) -> Dict:
    """Define the robot-specific mapping between feet frame names and indexes."""

    if robot != "iCubV2_5":
        raise Exception("Mapping between feet frame names and indexes only defined for iCubV2_5.")

    foot_name_to_index = {"l_sole": 53, "r_sole": 147}

    return foot_name_to_index

def compute_initial_joint_reference(robot: str) -> List:
    """Retrieve the robot-specific initial reference for the joints."""

    if robot != "iCubV2_5":
        raise Exception("Initial joint reference only defined for iCubV2_5.")

    initial_joint_reference = [0.0899, 0.0233, -0.0046, -0.5656, -0.3738, -0.0236,  # left leg
                               0.0899, 0.0233, -0.0046, -0.5656, -0.3738, -0.0236,  # right leg
                               0.1388792845, 0.0, 0.0,  # torso
                               -0.0629, 0.4397, 0.1825, 0.5387, # left arm
                               -0.0629, 0.4397, 0.1825, 0.5387] # right arm

    return initial_joint_reference