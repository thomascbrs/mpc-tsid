# coding: utf8

import numpy as np
import matplotlib.pylab as plt
import utils
import time

from TSID_Debug_controller_four_legs_fb_vel import controller, dt
import Safety_controller
import EmergencyStop_controller
import ForceMonitor
import MPC_Wrapper
import processing as proc
import MPC_Virtual
########################################################################
#                        Parameters definition                         #
########################################################################

# Time step
dt_mpc = 0.02
k_mpc = int(dt_mpc / dt)  # dt is dt_tsid, defined in the TSID controller script
t = 0.0  # Time
n_periods = 1  # Number of periods in the prediction horizon

# Simulation parameters
N_SIMULATION = 6500  # number of time steps simulated

# Initialize the error for the simulation time
time_error = False

# Lists to log the duration of 1 iteration of the MPC/TSID
t_list_tsid = [0] * int(N_SIMULATION)

# List to store the IDs of debug lines
ID_deb_lines = []

# Enable/Disable Gepetto viewer
enable_gepetto_viewer = True

# Create Joystick, ContactSequencer, FootstepPlanner, FootTrajectoryGenerator
# and MpcSolver objects
joystick, fstep_planner, logger, interface = utils.init_objects(dt, dt_mpc, N_SIMULATION, k_mpc, n_periods)

# Wrapper that makes the link with the solver that you want to use for the MPC
# First argument to True to have PA's MPC, to False to have Thomas's MPC
mpc_wrapper = MPC_Virtual.MPC_Virtual(True, dt_mpc, fstep_planner.n_steps, k_mpc, fstep_planner.T_gait)

# Enable/Disable hybrid control
enable_hybrid_control = True

########################################################################
#                            Gepetto viewer                            #
########################################################################

# Initialisation of the Gepetto viewer
solo = utils.init_viewer(enable_gepetto_viewer)

########################################################################
#                              PyBullet                                #
########################################################################

# Initialisation of the PyBullet simulator
pyb_sim = utils.pybullet_simulator(dt=0.001)

# Force monitor to display contact forces in PyBullet with red lines
myForceMonitor = ForceMonitor.ForceMonitor(pyb_sim.robotId, pyb_sim.planeId)

########################################################################
#                             Simulator                                #
########################################################################

# Define the default controller as well as emergency and safety controller
myController = controller(int(N_SIMULATION), k_mpc, n_periods)
mySafetyController = Safety_controller.controller_12dof()
myEmergencyStop = EmergencyStop_controller.controller_12dof()

for k in range(int(N_SIMULATION)):

    if (k % 1000) == 0:
        print("Iteration: ", k)

    # Process states update and joystick
    proc.process_states(solo, k, k_mpc, pyb_sim, interface, joystick, myController)

    # Process footstep planner
    proc.process_footsteps_planner(k, k_mpc, pyb_sim, interface, joystick, fstep_planner)

    # Process MPC once every k_mpc iterations of TSID
    if (k % k_mpc) == 0:
        f_applied = proc.process_mpc(k, k_mpc, interface, joystick, fstep_planner, mpc_wrapper,
                                     dt_mpc, ID_deb_lines)

    # Process Inverse Dynamics
    time_tsid = time.time()
    jointTorques = proc.process_invdyn(solo, k, f_applied, pyb_sim, interface, fstep_planner,
                                       myController, enable_hybrid_control)
    t_list_tsid[k] = time.time() - time_tsid  # Logging the time spent to run this iteration of inverse dynamics

    # Process PyBullet
    # proc.process_pybullet(pyb_sim, jointTorques)

    # Call logger object to log various parameters
    logger.call_log_functions(k, joystick, fstep_planner, interface, mpc_wrapper, myController,
                              False, pyb_sim.robotId, pyb_sim.planeId, solo)

####################
# END OF MAIN LOOP #
####################

print("END")

# Display what has been logged by the logger
logger.plot_graphs(enable_multiprocessing=False)

quit()

# Display duration of MPC block and Inverse Dynamics block
plt.figure()
plt.plot(t_list_tsid, 'k+')
plt.title("Time TSID")
plt.show(block=False)
