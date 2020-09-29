# coding: utf8
# coding: utf8

import sys
import os 
from sys import argv
sys.path.insert(0, os.getcwd()) # adds current directory to python path


import numpy as np
import matplotlib.pylab as plt
import utils
import time
from TSID_Debug_controller_four_legs_fb_vel import controller, dt
import MPC_Wrapper 
import FootstepPlanner
from crocoddyl_class.MPC_crocoddyl import MPC_crocoddyl


####################
# Recovery of Data
####################

folder_name = "log_eval/"
pathIn = "crocoddyl_eval/test_1/"
ddp_xs = np.load(pathIn + folder_name + "ddp_xs.npy")
ddp_us = np.load(pathIn + folder_name + "ddp_us.npy")
osqp_xs = np.load(pathIn + folder_name + "osqp_xs.npy")
osqp_us = np.load(pathIn + folder_name + "osqp_us.npy")
fsteps = np.load(pathIn + folder_name + "fsteps.npy")
xref = np.load(pathIn + folder_name + "xref.npy") 
l_feet = np.load(pathIn + folder_name + "l_feet.npy") 


####################
# Iteration 
####################

iteration = 250

dt_mpc = 0.02  # Time step of the MPC
k_mpc = int(dt_mpc / dt)  # dt is dt_tsid, defined in the TSID controller script
n_periods = 1  # Number of periods in the prediction horizon
T_gait = 0.64  # Duration of one gait period

# Create footstep planner object
fstep_planner = FootstepPlanner.FootstepPlanner(dt_mpc, n_periods, T_gait)
fstep_planner.xref = xref[:,:, iteration  ]
fstep_planner.fsteps = fsteps[:,:,iteration ]
fstep_planner.fsteps_mpc = fstep_planner.fsteps.copy()

######################################
#  Relaunch DDP to adjust the gains  #
######################################

Relaunch_DDP = True
mpc_wrapper_ddp_nl = MPC_crocoddyl( dt = dt_mpc , T_mpc = T_gait , mu = 0.9, inner = False, linearModel = False  , n_period = 1)
# mpc_wrapper_ddp_nl.shoulderWeights = 1000
# mpc_wrapper_ddp_nl.shoulder_hlim = 0.23
# # mpc_wrapper_ddp_nl.stateWeight[2] = 4.
# mpc_wrapper_ddp_nl.updateActionModel()
mpc_wrapper_ddp_nl_2 = MPC_crocoddyl( dt = dt_mpc , T_mpc = T_gait , mu = 0.9, inner = False, linearModel = False  , n_period = 1)

mpc_wrapper_ddp_nl.updateProblem(fstep_planner.fsteps , fstep_planner.xref )

mpc_wrapper_ddp_nl_2.updateProblem(fstep_planner.fsteps , fstep_planner.xref)

# Warm start : set candidate state and input vector           
us_osqp = osqp_us[:,:,iteration]
xs_osqp = osqp_xs[:,:,iteration]
u_init = []
x_init = []
x_init.append(fstep_planner.xref[:,0]) 
for j in range(len(us_osqp)) : 
    u_init.append(us_osqp[:,j])
    x_init.append(xs_osqp[:,j])

mpc_wrapper_ddp_nl.ddp.solve([] ,  [], 50 )

mpc_wrapper_ddp_nl_2.ddp.solve([] , [], 50 )


print(fstep_planner.xref)

#############
#  Plot     #
#############

# Predicted evolution of state variables
l_t = np.linspace(dt_mpc, n_periods*T_gait, np.int(n_periods*(T_gait/dt_mpc)))
print(l_t.shape)
l_str = ["X_osqp", "Y_osqp", "Z_osqp", "Roll_osqp", "Pitch_osqp", "Yaw_osqp", "Vx_osqp", "Vy_osqp", "Vz_osqp", "VRoll_osqp", "VPitch_osqp", "VYaw_osqp"]
l_str2 = ["X_ddp", "Y_ddp", "Z_ddp", "Roll_ddp", "Pitch_ddp", "Yaw_ddp", "Vx_ddp", "Vy_ddp", "Vz_ddp", "VRoll_ddp", "VPitch_ddp", "VYaw_ddp"]

index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
plt.figure()
for i in range(12):
    plt.subplot(3, 4, index[i])
    
    pl1, = plt.plot(l_t, ddp_xs[i,:,iteration], linewidth=2, marker='x')
    # pl2, = plt.plot(l_t, osqp_xs[i,:,iteration], linewidth=2, marker='x')

    if Relaunch_DDP : 
        pl3, = plt.plot(l_t, mpc_wrapper_ddp_nl.get_xrobot()[i,:], linewidth=2, marker='x')
        pl4, = plt.plot(l_t, mpc_wrapper_ddp_nl_2.get_xrobot()[i,:], linewidth=2, marker='x')
        # plt.legend([pl1,pl2,pl3,pl4] , [l_str2[i] , l_str[i], "ddp_redo" ,"no_w"])
        plt.legend([pl1,pl3,pl4] , [ l_str2[i],"nl" ,"nl 2"])


    
    else : 
        plt.legend([pl1,pl2] , [l_str2[i] , l_str[i] ])
    
mu = 0.9
# Desired evolution of contact forces
l_t = np.linspace(dt_mpc, n_periods*T_gait, np.int(n_periods*(T_gait/dt_mpc)))
l_str = ["FL_X_osqp", "FL_Y_osqp", "FL_Z_osqp", "FR_X_osqp", "FR_Y_osqp", "FR_Z_osqp", "HL_X_osqp", "HL_Y_osqp", "HL_Z_osqp", "HR_X_osqp", "HR_Y_osqp", "HR_Z_osqp"]
l_str2 = ["FL_X_ddp", "FL_Y_ddp", "FL_Z_ddp", "FR_X_ddp", "FR_Y_ddp", "FR_Z_ddp", "HL_X_ddp", "HL_Y_ddp", "HL_Z_ddp", "HR_X_ddp", "HR_Y_ddp", "HR_Z_ddp"]
index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
plt.figure()
for i in range(12):
    plt.subplot(3, 4, index[i])
    pl1, = plt.plot(l_t, ddp_us[i,:,iteration], linewidth=2, marker='x')
    # pl2, = plt.plot(l_t, osqp_us[i,:,iteration], linewidth=2, marker='x')

    # if i % 3*i == 0 :  
    #     print(i)
    #     xlim = []
    #     xlimm = []
    #     l_t2 = []
    #     for k in range(len(ddp_us[i,:,iteration])) : 
    #         if ddp_us[i+2,:,iteration][k] != 0 : 
    #             xlim.append(mu*ddp_us[i+2,:,iteration][k])
    #             xlimm.append(-mu*ddp_us[i+2,:,iteration][k])
    #             l_t2.append(l_t[k])
    #     plt.plot(l_t2 , xlim ,  "r--" )
    #     plt.plot(l_t2 , xlimm ,  "r--")
   
    if Relaunch_DDP : 
        pl3, = plt.plot(l_t, mpc_wrapper_ddp_nl.get_fpredicted()[i,:], linewidth=2, marker='x')
        pl4, = plt.plot(l_t, mpc_wrapper_ddp_nl_2.get_fpredicted()[i,:], linewidth=2, marker='x')
        # plt.legend([pl1,pl2,pl3,pl4] , [l_str2[i] , l_str[i], "ddp_redo" , "no_w" ])
        plt.legend([pl1,pl3,pl4] , [l_str2[i] , "nl" ,"nl 2"])

      
    else : 
        plt.legend([pl1,pl2] , [l_str2[i] , l_str[i] ])
    

plt.show(block=True)
