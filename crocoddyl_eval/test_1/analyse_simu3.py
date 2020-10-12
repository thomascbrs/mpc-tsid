# coding: utf8
# coding: utf8

import sys
import os 
from sys import argv
sys.path.insert(0, os.getcwd()) # adds current directory to python path


import numpy as np
import matplotlib.pylab as plt; plt.ion()
import utils
import time
from TSID_Debug_controller_four_legs_fb_vel import controller, dt
import MPC_Wrapper 
import FootstepPlanner

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
# l_feet = np.load(pathIn + folder_name + "l_feet.npy") 


####################
# Iteration 
####################

iteration = 15

dt_mpc = 0.02  # Time step of the MPC
k_mpc = int(dt_mpc / dt)  # dt is dt_tsid, defined in the TSID controller script
n_periods = 1  # Number of periods in the prediction horizon
T_gait = 0.64  # Duration of one gait period

xref[2,0, iteration  ] = xref[2,1, iteration  ]
xref[3:,0, iteration  ] = np.zeros(9)
print("\n\n")
print("x_init : " + str(xref[:6,0, iteration  ]))
print("x_ref : " + str(xref[:6,1, iteration  ]))
print("\n")

print("ddp1 --> ddp, gain of 0.01||f||^2")
print("ddp2 --> ddp, gain of 0.01||f-m*g/(nb contact)||^2 ")

# Create footstep planner object
fstep_planner = FootstepPlanner.FootstepPlanner(dt_mpc, n_periods, T_gait)
fstep_planner.xref = xref[:,:, iteration  ]
fstep_planner.fsteps = fsteps[:,:,iteration ]
fstep_planner.fsteps_mpc = fstep_planner.fsteps.copy()

######################################
#  Relaunch DDP 1                    #
######################################

enable_multiprocessing = False
mpc_wrapper_ddp_1 = MPC_Wrapper.MPC_Wrapper(False, dt_mpc, fstep_planner.n_steps,
                                        k_mpc, fstep_planner.T_gait, enable_multiprocessing)

mpc_wrapper_ddp_1.mpc.max_iteration = 10 # The solver is not warm started here
mpc_wrapper_ddp_1.relative_forces = True
mpc_wrapper_ddp_1.mpc.updateActionModel() #udpate
mpc_wrapper_ddp_1.solve(1,fstep_planner)

ddp1 = mpc_wrapper_ddp_1.mpc.ddp

######################################
#  Relaunch DDP 2         no gain     #
######################################

enable_multiprocessing = False
mpc_wrapper_ddp_2 = MPC_Wrapper.MPC_Wrapper(False, dt_mpc, fstep_planner.n_steps,
                                        k_mpc, fstep_planner.T_gait, enable_multiprocessing)

mpc_wrapper_ddp_2.mpc.max_iteration = 10 # The solver is not warm started here
mpc_wrapper_ddp_2.implicit_integration = False
mpc_wrapper_ddp_2.relative_forces = True
mpc_wrapper_ddp_2.mpc.updateActionModel() #udpate
mpc_wrapper_ddp_2.solve(1,fstep_planner)

ddp2 = mpc_wrapper_ddp_2.mpc.ddp

#############
#  Plot     #
#############

# Predicted evolution of state variables
l_t = np.linspace(dt_mpc, n_periods*T_gait, np.int(n_periods*(T_gait/dt_mpc)))

l_str = ["X_ddp", "Y_ddp", "Z_ddp", "Roll_ddp", "Pitch_ddp", "Yaw_ddp", "Vx_ddp", "Vy_ddp", "Vz_ddp", "VRoll_ddp", "VPitch_ddp", "VYaw_ddp"]

index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
plt.figure()
for i in range(12):
    plt.subplot(3, 4, index[i])
    
    
    pl1, = plt.plot(l_t, mpc_wrapper_ddp_1.mpc.get_xrobot()[i,:], linewidth=2, marker='x')
    pl2, = plt.plot(l_t, mpc_wrapper_ddp_2.mpc.get_xrobot()[i,:], linewidth=2, marker='x')
    plt.legend([pl1,pl2] , [l_str[i] , "ddp_rel" ])
    
    
mu = 0.9
# Desired evolution of contact forces
l_t = np.linspace(dt_mpc, n_periods*T_gait, np.int(n_periods*(T_gait/dt_mpc)))
l_str = ["FL_X_ddp", "FL_Y_ddp", "FL_Z_ddp", "FR_X_ddp", "FR_Y_ddp", "FR_Z_ddp", "HL_X_ddp", "HL_Y_ddp", "HL_Z_ddp", "HR_X_ddp", "HR_Y_ddp", "HR_Z_ddp"]
index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
plt.figure()
for i in range(12):
    plt.subplot(3, 4, index[i])

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
        
    pl1, = plt.plot(l_t, mpc_wrapper_ddp_1.mpc.get_fpredicted()[i,:], linewidth=2, marker='x')
    pl2, = plt.plot(l_t, mpc_wrapper_ddp_2.mpc.get_fpredicted()[i,:], linewidth=2, marker='x')
    plt.legend([pl1,pl2] , [l_str[i], "ddp_rel" ])

    

#plt.show(block=True)
