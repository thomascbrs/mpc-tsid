
# coding: utf8
# coding: utf8

import sys
import os 
from sys import argv
sys.path.insert(0, os.getcwd()) # adds current directory to python path

import random
import numpy as np
import matplotlib.pylab as plt; plt.ion()
# import matplotlib.pylab as plt
import utils
import time

####################
# Recovery of Data
####################

dt = 0.001

# Fstep planner
pathIn = "crocoddyl_eval/test_6/log_eval/behaviour/"
gait = np.load(pathIn +  "gait.npy"  , allow_pickle=True)
lC = np.load(pathIn +  "lC.npy"  , allow_pickle=True)
RPY = np.load(pathIn +  "RPY.npy" , allow_pickle=True)
lV = np.load(pathIn +  "lV.npy"  , allow_pickle=True)
lW = np.load(pathIn +  "lW.npy" , allow_pickle=True)

# No planner
pathIn = "crocoddyl_eval/test_6/log_eval/behaviour_dt/"
gait_dt = np.load(pathIn +  "gait.npy"  , allow_pickle=True)
lC_dt = np.load(pathIn +  "lC.npy" , allow_pickle=True)
RPY_dt = np.load(pathIn +  "RPY.npy"  , allow_pickle=True)
lV_dt = np.load(pathIn +  "lV.npy" , allow_pickle=True)
lW_dt = np.load(pathIn +  "lW.npy" , allow_pickle=True)

k_max_loop = lC.shape[1]
it_max = 2500

t_range = np.array([k*dt for k in range(it_max)])


plt.figure()

index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
lgd = ["Pos CoM X [m]", "Pos CoM Y [m]", "Pos CoM Z [m]", "Roll [deg]", "Pitch [deg]", "Yaw [deg]",
        "Lin Vel CoM X [m/s]", "Lin Vel CoM Y [m/s]", "Lin Vel CoM Z [m/s]", "Ang Vel Roll [deg/s]", "Ang Vel Pitch [deg/s]", "Ang Vel Yaw [deg/s]"]

lwidth = 1.5
for i in range(12):
    plt.subplot(3, 4, index[i])
    if i < 3:
        plt.plot(t_range, lC[i, :it_max], "k", linewidth= lwidth)
        plt.plot(t_range, lC_dt[i, :it_max], "r--", linewidth= lwidth )
    elif i < 6:
        plt.plot(t_range, (180/3.1415)*RPY[i-3, :it_max], "k", linewidth= lwidth)
        plt.plot(t_range, (180/3.1415)*RPY_dt[i-3, :it_max], "r--", linewidth= lwidth)
        
    elif i < 9:
        plt.plot(t_range, lV[i-6, :it_max], "k", linewidth= lwidth)
        plt.plot(t_range, lV_dt[i-6, :it_max], "r--", linewidth= lwidth )
    else:
        plt.plot(t_range, (180/3.1415)*lW[i-9, :it_max], "k", linewidth= lwidth)
        plt.plot(t_range, (180/3.1415)*lW_dt[i-9, :it_max], "r--", linewidth= lwidth)

    plt.xlabel("Time [s]")
    plt.ylabel(lgd[i])
    if index[i] == 2 : 
        plt.legend(["footstep", "period optim"] ,ncol = 2,  bbox_to_anchor=(0, 1),
              loc='lower left')

    # plt.legend(["fstep", "optim dt"])


plt.suptitle("State of the robot in TSID Vs PyBullet Vs Reference (local frame)")



#####"" bar plot
k_mpc = 10
dt_mpc = 0.01
gait_bar = np.zeros((int(it_max/k_mpc) , 4)) 
gait_bar_dt = np.zeros((int(it_max/k_mpc) , 4)) 
for k in range(int(it_max/k_mpc)) : 
    gait_bar[k,:] = gait[0,1:,k]
    gait_bar_dt[k,:] = gait_dt[0,1:,k]

fig, ax = plt.subplots()      
for j in range(gait_bar.shape[0]) : # row    
    line = gait_bar[j,:]
    line_dt = gait_bar_dt[j,:]
    cl = []
    if line[0] == 1  :
        cl.append("k")
    else : 
        cl.append("w")

    if line[2] == 1  :
        cl.append("k")
    else : 
        cl.append("w")
    
    if line_dt[0] == 1  :
        cl.append("k")
    else : 
        cl.append("w")
        
    if line_dt[2] == 1  :
        cl.append("k")
    else : 
        cl.append("w")


    start = (j)*dt_mpc
    plt.barh([6,5,3,2] , 4*[dt_mpc] , color = cl , left = start )

y_pos = [6,5,3,2]
ax.set_yticks(y_pos)
ax.set_yticklabels(["LH","LF","       LH    \nOptim period","       RH    \nOptim period"])
plt.axvline(x = 55*dt_mpc , color = "k" , linestyle = "--")
ax.set_xlim([0.5,1])
plt.xlabel("Time [s]")
plt.legend( ["$V_y$ perturbation", "Foot on the ground"] ,ncol = 1,  bbox_to_anchor=(0, 1),
              loc='lower left')


    

gait_bar = np.zeros((int(it_max/k_mpc) , 4)) 
for k in range(int(it_max/k_mpc)) : 
    gait_bar[k,:] = gait_dt[0,1:,k]

fig, ax = plt.subplots()      
plt.suptitle("Gait Matrix DT")
for j in range(gait_bar.shape[0]) : # row    
    line = gait_bar[j,:]
    if line[0] == 1 and line[1] == 1 :
        cl = ["k","k","k","k"]
    elif line[0] == 1 and line[1] == 0  :
        cl = ["k","w","w","k"]
    else : 
        cl = ["w" , "k" , "k" , "w"]
    start = (j)*dt_mpc
    plt.barh([3,2,4,1] , 4*[dt_mpc] , color = cl , left = start )

y_pos = [4,3,2,1]
ax.set_yticks(y_pos)
ax.set_yticklabels(["LH","LF","RF","RH"])
pl1 = plt.axvline(x = 55*dt_mpc , color = "k" , linestyle = "--")
ax.set_xlim([0.5,1.2])



    

