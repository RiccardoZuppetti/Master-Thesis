# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import os
import time
import yarp
import argparse
import numpy as np
from adherent.trajectory_control import trajectory_controller
from adherent.trajectory_control.utils import define_foot_name_to_index_mapping
from adherent.trajectory_control.utils import compute_initial_joint_reference
from adherent.data_processing.utils import define_feet_frames_and_links
import bipedal_locomotion_framework as blf

# ==================
# USER CONFIGURATION
# ==================

parser = argparse.ArgumentParser()

parser.add_argument("--trajectory_path", help="Path where the generated trajectory is stored. Relative path from script folder.",
                    type = str, default = "../datasets/inference/")
parser.add_argument("--time_scaling", help="Time scaling to be applied to the generated trajectory. Keep it integer.",
                    type=int, default=2)
parser.add_argument("--footstep_scaling", help="Footstep scaling to be applied to the generated footsteps. Keep it between 0 and 1.",
                    type=float, default=0.5)
parser.add_argument("--deactivate_postural", help="Deactivate usage of the postural from Adherent.", action="store_true")
parser.add_argument("--real_robot", help="Use the real iCub.", action="store_true")

args = parser.parse_args()

trajectory_path = args.trajectory_path
time_scaling = args.time_scaling
footstep_scaling = args.footstep_scaling
use_joint_references = not args.deactivate_postural
real_robot = args.real_robot

# Debug
print("REAL ROBOT:", real_robot)
input("Press Enter to continue")

# ==================
# YARP CONFIGURATION
# ==================

# Initialize the Yarp clock
if real_robot:
    yarp.Network.init()
else:
    yarp.Network.init(yarp.YARP_CLOCK_NETWORK)

# Open port for the robot logger
if real_robot:
    port = blf.yarp_utilities.BufferedPortVectorsCollection()
    port.open("/adherent/logger/data:o")

    # Debug
    # print("port open")
    # for i in range(10):
    #     data = port.prepare()
    #     v1 = np.random.rand(4)
    #     v2 = np.random.rand(4)
    #     data.vectors = {"v1": v1, "v2": v2}
    #     port.write()
    #     time.sleep(0.1) # Required not to be too fast
    # port.close()
    # print("port closed")
    # input()export $YARP_ROBOT_NAME=iCubGenova09


# ===================================
# TRAJECTORY CONTROLLER CONFIGURATION
# ===================================

# Retrieve script directory
script_directory = os.path.dirname(os.path.abspath(__file__))

# Retrieve the robot model
robot_urdf = "/iit/sources/robotology-superbuild/src/icub-models/iCub/robots/iCubGenova09/model.urdf"

# Define the paths for the generated footsteps and postural
trajectory_path = os.path.join(script_directory, trajectory_path)
footsteps_path = trajectory_path + "footsteps.txt"
posturals_path = trajectory_path + "postural.txt"

# Define the beginning of the path where the trajectory control data will be stored
storage_path = os.path.join(script_directory, "../datasets/trajectory_control_simulation/sim_")

# Define the beginning of the path where the trajectory control data will be stored
if real_robot:
    storage_path = os.path.join(script_directory, "../datasets/trajectory_control_real_robot/exp_")
else:
    storage_path = os.path.join(script_directory, "../datasets/trajectory_control_simulation/sim_")

# Define the joints list used by the different components in the pipeline
controlled_joints = ['l_hip_pitch', 'l_hip_roll', 'l_hip_yaw', 'l_knee', 'l_ankle_pitch', 'l_ankle_roll',  # left leg
                     'r_hip_pitch', 'r_hip_roll', 'r_hip_yaw', 'r_knee', 'r_ankle_pitch', 'r_ankle_roll',  # right leg
                     'torso_pitch', 'torso_roll', 'torso_yaw',  # torso
                     'l_shoulder_pitch', 'l_shoulder_roll', 'l_shoulder_yaw', 'l_elbow', # left arm
                     'r_shoulder_pitch', 'r_shoulder_roll', 'r_shoulder_yaw', 'r_elbow'] # right arm

# Check that the l_sole and r_sole frames indexes in the model are the correct ones
# import idyntree.bindings as idt
# mdl_loader = idt.ModelLoader()
# mdl_loader.loadReducedModelFromFile(robot_urdf, controlled_joints)
# model = mdl_loader.model()
# print("***************")
# print("l_sole", model.getFrameIndex(frameName="l_sole"))
# print("r_sole", model.getFrameIndex(frameName="r_sole"))
# input("If the frames indexes are correct, press Enter to continue")

# Define robot-specific feet mapping between feet frame names and indexes
foot_name_to_index = define_foot_name_to_index_mapping(robot="iCubV3")

# Define robot-specific initial joint reference
initial_joint_reference = compute_initial_joint_reference(robot="iCubV3")

# Define robot-specific feet frames and links
feet_frames, feet_links = define_feet_frames_and_links(robot="iCubV3")

# Instantiate the trajectory controller
controller = trajectory_controller.TrajectoryController.build(robot_urdf=robot_urdf,
                                                              footsteps_path=footsteps_path,
                                                              posturals_path=posturals_path,
                                                              feet_frames=feet_frames,
                                                              storage_path=storage_path,
                                                              time_scaling=time_scaling,
                                                              footstep_scaling=footstep_scaling,
                                                              use_joint_references=use_joint_references,
                                                              controlled_joints=controlled_joints,
                                                              foot_name_to_index=foot_name_to_index,
                                                              initial_joint_reference=initial_joint_reference,
                                                              real_robot=real_robot)

if real_robot:

    # # OPEN LOOP
    # k_zmp = 0.0
    # k_dcm = 0.0
    # k_com = 0.0

    # TODO: tune the controller gains for the real robot
    # CLOSED LOOP
    k_zmp = 1.8 # 1.8 walking controller
    k_dcm = 1.1 # 0.0 walking controller
    k_com = 5.0 # 5.0 walking controller

else:

    k_zmp = 1.0
    k_dcm = 1.1
    k_com = 4.0

# Debug
print("gains - k_zmp: ",k_zmp,"k_com: ",k_com,"k_dcm: ",k_dcm)
input("If the gains are ok, press Enter to continue")

# Configure all the components of the trajectory control pipeline
controller.configure(k_zmp=k_zmp, k_com=k_com, k_dcm=k_dcm)

# ===================
# TRAJECTORY PLANNING
# ===================

# Trajectory optimization
controller.compute_dcm_trajectory()

# ==================
# TRAJECTORY CONTROL
# ==================

# Trajectory control loop running at dt = 100 Hz
for idx in np.arange(start=0, stop=controller.get_trajectory_duration(), step=controller.get_dt()):

    # Measure joint values and feet wrenches
    controller.read_data()

    # Update the legged odometry estimator
    controller.update_legged_odom()

    # Advance the DCM and swing foot planners
    controller.update_planners()

    # Compute the desired CoM
    controller.update_controllers()

    # Update the feet, CoM and joint targets for the inverse kinematics
    controller.update_ik_targets(idx)

    # Compute the joint reference realizing the ik targets
    controller.retrieve_joint_reference()

    # Set the joint reference
    controller.set_current_joint_reference(idx) # TODO: comment to run the script without sending the references
    # print("References not sent") # TODO: uncomment if you run the script without sending the references

    # Update the storage of the quantities of interest
    controller.update_storage(idx)

    if real_robot:
        # Send data to the robot-logger
        controller.send_data_to_logger(port)

# At the end of the control loop, store the relevant data
controller.storage.save_data_as_json()

# Close port for robot logger
if real_robot:
    port.close()

