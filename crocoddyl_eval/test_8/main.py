# coding: utf8
import sys
import os 
from sys import argv
sys.path.insert(0, os.getcwd()) # adds current directory to python path

import numpy as np
import utils
import time

from TSID_Debug_controller_four_legs_fb_vel import controller, dt
import Safety_controller
import EmergencyStop_controller
import ForceMonitor
import processing as proc
import MPC_Wrapper
import pybullet as pyb
import Logger
from crocoddyl_class.MPC_crocoddyl import MPC_crocoddyl
from crocoddyl_class.MPC_crocoddyl_planner import *
from crocoddyl_class.MPC_crocoddyl_planner_time import *



def run_scenario(envID, velID, dt_mpc, k_mpc, t, n_periods, T_gait, N_SIMULATION, type_MPC, pyb_feedback , mpc_model , desired_speed):

    ########################################################################
    #                        Parameters definition                         #
    ########################################################################
    """# Time step
    dt_mpc = 0.02
    k_mpc = int(dt_mpc / dt)  # dt is dt_tsid, defined in the TSID controller script
    t = 0.0  # Time
    n_periods = 1  # Number of periods in the prediction horizon
    T_gait = 0.48  # Duration of one gait period
    # Simulation parameters
    N_SIMULATION = 1000  # number of time steps simulated"""

    N_1 = np.round(int(np.around(abs(desired_speed[0]), decimals = 1 ) * 10000 + 10000) , -3 )
    N_2 = np.round(int(np.around(abs(desired_speed[5]), decimals = 1 ) * 2500 + 10000) , -3 )
    N_3 = np.round(int(np.around(abs(desired_speed[1]), decimals = 1 ) * 10000 + 10000) , -3 )
    N_SIMULATION = max(N_1 , N_2 , N_3)
    N_SIMULATION = 2000


    # to generate trajectory 
    torques_ff = np.zeros((12, N_SIMULATION))
    torques_pd = np.zeros((12, N_SIMULATION))
    torques_sent = np.zeros((12, N_SIMULATION))

    qtsid = np.zeros((19,N_SIMULATION))
    vtsid = np.zeros((18, N_SIMULATION))

    # Initialize the error for the simulation time
    time_error = False

    # Lists to log the duration of 1 iteration of the MPC/TSID
    t_list_tsid = [0] * int(N_SIMULATION)
    t_list_loop = [0] * int(N_SIMULATION)
    t_list_mpc = [0] * int(N_SIMULATION)

    # List to store the IDs of debug lines
    ID_deb_lines = []

    # Enable/Disable Gepetto viewer
    enable_gepetto_viewer = True

    # Which MPC solver you want to use
    # True to have PA's MPC, to False to have Thomas's MPC
    """type_MPC = True"""

    # Create Joystick, FootstepPlanner, Logger and Interface objects
    joystick, fstep_planner, logger_ddp, interface = utils.init_objects(
        dt, dt_mpc, N_SIMULATION, k_mpc, n_periods, T_gait, False)
    
     # Multi simulation environment
    joystick.multi_simu = True
    joystick.Vx_ref = desired_speed[0]
    joystick.Vy_ref = desired_speed[1]
    joystick.Vw_ref = desired_speed[5]

    # Multi simulation environment

    # Create a new logger type for the second solver
    logger_osqp = Logger.Logger(N_SIMULATION, dt, dt_mpc, k_mpc, n_periods, T_gait, True)

    # Wrapper that makes the link with the solver that you want to use for the MPC
    # First argument to True to have PA's MPC, to False to have Thomas's MPC
    enable_multiprocessing = False
    # Initialize the two algorithms
    mpc_wrapper_ddp = MPC_Wrapper.MPC_Wrapper(False, dt_mpc, fstep_planner.n_steps,
                                          k_mpc, fstep_planner.T_gait, enable_multiprocessing)

    mpc_wrapper_osqp = MPC_Wrapper.MPC_Wrapper(True, dt_mpc, fstep_planner.n_steps,
                                          k_mpc, fstep_planner.T_gait, enable_multiprocessing)
                        
    
    mpc_wrapper_ddp_nl = MPC_crocoddyl( dt = dt_mpc , T_mpc = T_gait , mu = 0.9, inner = False, linearModel = False  , n_period = 1)

    # MPC with augmented states
    mpc_planner = MPC_crocoddyl_planner(dt = dt_mpc , T_mpc = fstep_planner.T_gait , n_periods = n_periods)

    mpc_planner_time = MPC_crocoddyl_planner_time(dt = dt_mpc , T_mpc = fstep_planner.T_gait, n_periods = n_periods , min_fz = 1)


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
    pyb_sim = utils.pybullet_simulator(envID, dt=0.001)

    # Force monitor to display contact forces in PyBullet with red lines
    myForceMonitor = ForceMonitor.ForceMonitor(pyb_sim.robotId, pyb_sim.planeId)

    ########################################################################
    #                             Simulator                                #
    ########################################################################

    # Define the default controller as well as emergency and safety controller
    myController = controller(int(N_SIMULATION), k_mpc, n_periods, T_gait)
    mySafetyController = Safety_controller.controller_12dof()
    myEmergencyStop = EmergencyStop_controller.controller_12dof()

    import matplotlib.pylab as plt

    for k in range(int(N_SIMULATION)):
        time_loop = time.time()

        if (k % 1000) == 0:
            print("Iteration: ", k)

        # Process states update and joystick
        proc.process_states(solo, k, k_mpc, velID, pyb_sim, interface, joystick, myController, pyb_feedback)

        if np.isnan(interface.lC[2, 0]):
            print("NaN value for the position of the center of mass. Simulation likely crashed. Ending loop.")
            break

        # Process footstep planner
        proc.process_footsteps_planner(k, k_mpc, pyb_sim, interface, joystick, fstep_planner)       
        

        if (k % k_mpc) == 0:
            print(fstep_planner.xref[6,0:2])
            time_mpc = time.time()
            # Run both algorithms
            # proc.process_mpc(k, k_mpc, interface, joystick, fstep_planner, mpc_wrapper_osqp,
            #                  dt_mpc, ID_deb_lines)
            if mpc_model == 0 : 
                proc.process_mpc(k, k_mpc, interface, joystick, fstep_planner, mpc_wrapper_ddp,
                             dt_mpc, ID_deb_lines)
            else : 
                fstep_planner.getRefStates((k/k_mpc), fstep_planner.T_gait, interface.lC, interface.abg,
                    interface.lV, interface.lW, joystick.v_ref, h_ref=0.2027682)
            
            if mpc_model == 1 : 
                mpc_wrapper_ddp_nl.updateProblem(fstep_planner.fsteps , fstep_planner.xref )
                mpc_wrapper_ddp_nl.ddp.solve([], [], 10 )

            if mpc_model == 2 : 
                mpc_planner.solve(k, fstep_planner.xref , interface.l_feet , interface.oMl )
            
            if mpc_model == 3 : 
                mpc_planner_time.solve(k, fstep_planner.xref , interface )
            
            logger_ddp.log_fstep_planner( k , fstep_planner)
           
            t_list_mpc[k] = time.time() - time_mpc            
            print( int(k/k_mpc))


        # Foot placement : 
        if mpc_model == 2 : 
            fstep_planner.fsteps_invdyn = mpc_planner.fsteps.copy()
        if mpc_model == 3 :   
            fstep_planner.fsteps_invdyn = mpc_planner_time.fsteps.copy()
    

        if mpc_model == 1 : 
            f_applied = mpc_wrapper_ddp_nl.get_latest_result()
        elif mpc_model == 2 : 
            f_applied = mpc_planner.get_latest_result()
        elif mpc_model == 3 : 
            f_applied = mpc_planner_time.get_latest_result()
        else : 
            f_applied = mpc_wrapper_ddp.get_latest_result()


        # Process Inverse Dynamics
        time_tsid = time.time()
        jointTorques = proc.process_invdyn(solo, k, f_applied, pyb_sim, interface, fstep_planner,
                                           myController, enable_hybrid_control)
        t_list_tsid[k] = time.time() - time_tsid  # Logging the time spent to run this iteration of inverse dynamics

        # Process PD+ (feedforward torques and feedback torques)
        for i_step in range(1):

            # Process the PD+
            jointTorques = proc.process_pdp(pyb_sim, myController)

            if myController.error:
                print('NaN value in feedforward torque. Ending loop.')
                break

            # Process PyBullet
            proc.process_pybullet(pyb_sim, k, envID, jointTorques)

        # Call logger object to log various parameters
        # logger_ddp.call_log_functions(k, pyb_sim, joystick, fstep_planner, interface, mpc_wrapper_ddp, myController,
        #                          False, pyb_sim.robotId, pyb_sim.planeId, solo)
        
        # if (k % k_mpc) == 0:
        #     # logger_ddp.log_fstep_planner( k , fstep_planner)
        #     logger_osqp.log_predicted_trajectories(k, mpc_wrapper_osqp)

        t_list_loop[k] = time.time() - time_loop

 


        #######
        # Generate traj : 
        torques_ff[:, k:(k+1)] =  np.reshape(myController.tau_ff ,(12,1))
        torques_pd[:, k:(k+1)] = np.reshape(myController.tau_pd ,(12,1))
        torques_sent[:, k:(k+1)] = np.reshape(myController.tau ,(12,1))
        qtsid[:,k:(k+1)] = np.reshape(myController.qtsid ,(19,1))
        vtsid[:,k:(k+1)] = np.reshape(myController.vtsid ,(18,1))

        #########################
        #   Camera
        #########################

        # if (k % 2) == 0:
        #     img = pyb.getCameraImage(width=int(1.2*1920), height=int(1.2*1080), renderer=pyb.ER_BULLET_HARDWARE_OPENGL)
        #     #if k == 0:
        #     #    newpath = r'/tmp/recording'
        #     #    if not os.path.exists(newpath):
        #     #       os.makedirs(newpath)
        #     if (int(k/2) < 10):
        #         plt.imsave('tmp/recording/frame_00'+str(int(k/2))+'.png', img[2])
        #     elif int(k/2) < 100:
        #         plt.imsave('tmp/recording/frame_0'+str(int(k/2))+'.png', img[2])
        #     else:
        #         plt.imsave('tmp/recording/frame_'+str(int(k/2))+'.png', img[2])

    ####################
    # END OF MAIN LOOP #
    ####################

    finished = False 

    if k == N_SIMULATION - 1 : 
        finished = True 
    
    print(finished)

    print("END")

    pyb.disconnect()

    return torques_ff , torques_pd , torques_sent, qtsid , vtsid


"""# Display what has been logged by the logger
logger.plot_graphs(enable_multiprocessing=False)

quit()

# Display duration of MPC block and Inverse Dynamics block
plt.figure()
plt.plot(t_list_tsid, 'k+')
plt.title("Time TSID")
plt.show(block=True)"""
