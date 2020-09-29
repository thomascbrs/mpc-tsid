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
dt_mpc = 0.02
k_mpc = int(dt_mpc / dt)


folder_name = "log_eval/"
pathIn = "crocoddyl_eval/test_1/"
ddp_xs = np.load(pathIn + folder_name + "ddp_xs.npy")
ddp_us = np.load(pathIn + folder_name + "ddp_us.npy")
osqp_xs = np.load(pathIn + folder_name + "osqp_xs.npy")
osqp_us = np.load(pathIn + folder_name + "osqp_us.npy")
fsteps = np.load(pathIn + folder_name + "fsteps.npy")
xref = np.load(pathIn + folder_name + "xref.npy") 
o_feet = np.load(pathIn + folder_name + "o_feet.npy") 
oC = np.load(pathIn + folder_name + "oC.npy") 
o_shoulders = np.load(pathIn + folder_name + "o_shoulders.npy") 

####################
# Iteration 
####################
N_iterations = xref.shape[2]
gait = np.zeros((20, 5))
print(N_iterations)

distance = np.zeros((4,N_iterations))
distance_2 = np.zeros((4,N_iterations))
l_shoulders = np.array([[0.19, 0.19, -0.19, -0.19], [0.15005, -0.15005,
                                                                  0.15005, -0.15005], [0.0, 0.0, 0.0, 0.0]])

for i in range(N_iterations) :    
    index = next((idx for idx, val in np.ndenumerate(fsteps[:, 0 , i]) if val==0.0), 0.0)[0]
    gait[:, 0] = fsteps[:, 0 , i]
    gait[:index, 1:] = 1.0 - (np.isnan(fsteps[:index, 1::3,i]) | (fsteps[:index, 1::3,i] == 0.0))
    fsteps[np.isnan(fsteps[:,:,i]) , i] = 0.0  

    for j in range(4) : 
        if gait[0,j+1] != 0 :
            dx = o_feet[0,j,i*k_mpc] - o_shoulders[0,j,i*k_mpc] 
            dy = o_feet[1,j,i*k_mpc] - o_shoulders[1,j,i*k_mpc] 
            dz = xref[2,0,i] - l_shoulders[0,j]*np.sin(xref[4,0,i]) + l_shoulders[1,j]*np.cos(xref[4,0,i])*np.sin(xref[3,0,i])  
            # print( dz )
            x = xref[:,0,i]
            lever_arms =  np.reshape(fsteps[0, 1:,i], (3, 4), order='F')

            psh = np.array( [x[0] + l_shoulders[0,j] - l_shoulders[1,j]*x[5] - lever_arms[0,j], 
                            x[1] + l_shoulders[1,j] + l_shoulders[0,j]*x[5] - lever_arms[1,j], 
                            x[2] + l_shoulders[1,j]*x[3] - l_shoulders[0,j]*x[4] ] ) 

            distance[j,i] = np.sqrt(dx**2 + dy**2 + dz**2)
            distance_2[j,i] = np.sqrt(psh[0]**2 + psh[1]**2 + psh[2]**2)


    
l_t = np.linspace(0,N_iterations*dt_mpc , N_iterations)

plt.figure()
for i in range(4) : 
    plt.subplot(2,2,i+1)
    plt.plot(l_t , distance[i,:] , "x")
    plt.plot(l_t , distance_2[i,:] , "x")
    
print(np.max(distance))
    

plt.show(block=True)
