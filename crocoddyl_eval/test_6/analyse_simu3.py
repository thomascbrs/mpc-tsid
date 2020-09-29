# coding: utf8
# coding: utf8

import sys
import os 
from sys import argv
sys.path.insert(0, os.getcwd()) # adds current directory to python path

import random
import numpy as np
import matplotlib.pylab as plt
import utils
import time
from TSID_Debug_controller_four_legs_fb_vel import controller, dt
import MPC_Wrapper 
import FootstepPlanner
from crocoddyl_class.MPC_crocoddyl_planner_time import *
from crocoddyl_class.MPC_crocoddyl_planner import *
####################
# Recovery of Data
####################

folder_name = "log_eval/"
pathIn = "crocoddyl_eval/test_6/"
oC = np.load(pathIn + folder_name + "oC.npy" , allow_pickle=True )
o_feet_ = np.load(pathIn + folder_name + "o_feet_.npy" , allow_pickle=True ) # global position of the feet
o_feet_heur = np.load(pathIn + folder_name + "o_feet_heur.npy" , allow_pickle=True )
gait_ = np.load(pathIn + folder_name + "gait_.npy" , allow_pickle=True )
ddp_xs = np.load(pathIn + folder_name + "pred_trajectories.npy" , allow_pickle=True )
ddp_us = np.load(pathIn + folder_name + "pred_forces.npy" , allow_pickle=True )
l_feet_ = np.load(pathIn + folder_name + "l_feet_.npy" , allow_pickle=True ) # Local position of the feet
xref = np.load(pathIn + folder_name + "xref.npy" , allow_pickle=True ) 
####################
# Iteration 
####################

iteration = 200     
dt_mpc = 0.02  # Time step of the MPC
k_mpc = int(dt_mpc / dt)  # dt is dt_tsid, defined in the TSID controller script
n_periods = 1  # Number of periods in the prediction horizon
T_gait = 0.48  # Duration of one gait period

###################
# Parameters 
###################

gait = gait_[:,:,iteration]
l_feet = l_feet_[:,:,iteration] 
print(gait)

mpc_planner = MPC_crocoddyl_planner(dt = dt_mpc , T_mpc = T_gait , n_periods = n_periods)    

    
for i in range(iteration  +  1) :
    if i == 0 : 
        mpc_planner.solve(0, xref[:,:, i] , o_feet_[:,:,0 ] )
    else : 
        mpc_planner.solve(i, xref[:,:, i] , o_feet_[:,:,i-1 ] )

print(mpc_planner.ddp.cost)
print(mpc_planner.ddp.iter)

mpc_planner_2 = MPC_crocoddyl_planner(dt = dt_mpc , T_mpc = T_gait , n_periods = n_periods)    

for i in range(iteration  +  1) :
    if i == 0 : 
        mpc_planner_2.solve(0, xref[:,:, i] , o_feet_[:,:,0 ] )
    else : 
        mpc_planner_2.updateProblem(i , xref[:,:, i]  , o_feet_[:,:,i-1 ] )

        
        for index,elt in enumerate(mpc_planner_2.ListAction) : 
            if elt.__class__.__name__ == "ActionModelQuadrupedStep" :  
                elt.heuristicWeights =  np.zeros(8)
                # elt.stateWeights =  np.zeros(12) 
                mpc_planner_2.ListAction[index-1].heuristicWeights = np.sqrt(2)*mpc_planner_2.heuristicWeights
                # mpc_planner_2.ListAction[index+1].stateWeights = np.sqrt(2)*mpc_planner_2.stateWeights

        # Solve problem
        mpc_planner_2.ddp.solve(mpc_planner_2.x_init,mpc_planner_2.u_init, mpc_planner_2.max_iteration)        
        
        # Get the results
        mpc_planner_2.get_fsteps()




print(mpc_planner_2.ddp.cost)
print(mpc_planner_2.ddp.iter)

print("same input")

cost = 0
cost_2 = 0
for index,elt in enumerate(mpc_planner.ListAction) :
    data = elt.createData()
    elt.calc(data, mpc_planner.ddp.xs[index] , mpc_planner.ddp.us[index] )
    cost += data.cost       
    print(elt.__class__.__name__)
    print("cost 1 : " , data.cost)

    
    data = mpc_planner_2.ListAction[index].createData()
    mpc_planner_2.ListAction[index].calc(data, mpc_planner.ddp.xs[index] , mpc_planner.ddp.us[index] )
    cost_2 += data.cost
    print("cost 2 : " , data.cost   )


data = mpc_planner_2.terminalModel.createData()
mpc_planner_2.terminalModel.calc(data, mpc_planner.ddp.xs[-1] , mpc_planner.ddp.us[index] )
cost_2 += data.cost

data = mpc_planner.terminalModel.createData()
mpc_planner.terminalModel.calc(data, mpc_planner.ddp.xs[-1] , mpc_planner.ddp.us[index] )
cost += data.cost

print(cost)
print(cost_2)

#############
#  Plot     #
#############

# Predicted evolution of state variables
l_t = np.linspace(0., T_gait , np.int(T_gait/dt_mpc)*n_periods)

l_str2 = ["X_ddp", "Y_ddp", "Z_ddp", "Roll_ddp", "Pitch_ddp", "Yaw_ddp", "Vx_ddp", "Vy_ddp", "Vz_ddp", "VRoll_ddp", "VPitch_ddp", "VYaw_ddp"]

index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
plt.figure()
for i in range(12):
    plt.subplot(3, 4, index[i])       


    pl1, = plt.plot(l_t, mpc_planner.Xs[i,:], linewidth=2, marker='x')
    pl2, = plt.plot(l_t, mpc_planner_2.Xs[i,:], linewidth=2, marker='x')
    plt.legend([pl1,pl2] , [l_str2[i] ,  "ddp_time" ])
       

    

# Desired evolution of contact forces
l_str2 = ["FL_X_ddp", "FL_Y_ddp", "FL_Z_ddp", "FR_X_ddp", "FR_Y_ddp", "FR_Z_ddp", "HL_X_ddp", "HL_Y_ddp", "HL_Z_ddp", "HR_X_ddp", "HR_Y_ddp", "HR_Z_ddp"]
index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
plt.figure()
for i in range(12):
    plt.subplot(3, 4, index[i])    
   
  
    pl1, = plt.plot(l_t,mpc_planner.Us[i,:], linewidth=2, marker='x')
    pl2, = plt.plot(l_t,mpc_planner_2.Us[i,:], linewidth=2, marker='x')
    plt.legend([pl1,pl2] , [l_str2[i] , "ddp_time" ])
    

plt.figure()

index = next((idx for idx, val in np.ndenumerate(mpc_planner.gait[:, 0]) if val==0.0), 0.0)[0]

# Feet on the ground at the beginning : k - o
# inital position for optimisation :   k - x
for i in range(4) : 
    if mpc_planner.gait[0,i+1] == 1 : 
        plt.plot(mpc_planner.fsteps[0,3*i+1] , mpc_planner.fsteps[0,3*i+2] , "ko" , markerSize = 7   )
    else : 
        plt.plot(mpc_planner.p0[2*i] , mpc_planner.p0[2*i+1] , "kx" , markerSize = 8   )

# Position of the center of mass
for elt in mpc_planner.Xs : 
    plt.plot(mpc_planner.Xs[0,:] ,mpc_planner.Xs[1,:] , "gx")


for i in range(4) : 
    for k in range(1,index) : 
        if mpc_planner.fsteps[k,3*i+1] != 0. and mpc_planner.gait[k,i+1] !=0 and mpc_planner.gait[k-1,i+1] == 0  : 
            if i == 0 :
                plt.plot(mpc_planner.fsteps[k,3*i+1] , mpc_planner.fsteps[k,3*i+2] , "bo" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none'  )
            if i == 1 :
                plt.plot(mpc_planner.fsteps[k,3*i+1] , mpc_planner.fsteps[k,3*i+2] , "ro" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none' )
            if i == 2 :
                plt.plot(mpc_planner.fsteps[k,3*i+1] , mpc_planner.fsteps[k,3*i+2] , "ko" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none')
            if i == 3 :
                plt.plot(mpc_planner.fsteps[k,3*i+1] , mpc_planner.fsteps[k,3*i+2] , "go" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none')


plt.figure()

index = next((idx for idx, val in np.ndenumerate(mpc_planner_2.gait[:, 0]) if val==0.0), 0.0)[0]

# Feet on the ground at the beginning : k - o
# inital position for optimisation :   k - x
for i in range(4) : 
    if mpc_planner_2.gait[0,i+1] == 1 : 
        plt.plot(mpc_planner_2.fsteps[0,3*i+1] , mpc_planner_2.fsteps[0,3*i+2] , "ko" , markerSize = 7   )
    else : 
        plt.plot(mpc_planner_2.p0[2*i] , mpc_planner_2.p0[2*i+1] , "kx" , markerSize = 8   )

# Position of the center of mass
for elt in mpc_planner_2.Xs : 
    plt.plot(mpc_planner_2.Xs[0,:] ,mpc_planner_2.Xs[1,:] , "gx")


for i in range(4) : 
    for k in range(1,index) : 
        if mpc_planner_2.fsteps[k,3*i+1] != 0. and mpc_planner_2.gait[k,i+1] !=0 and mpc_planner_2.gait[k-1,i+1] == 0  : 
            if i == 0 :
                plt.plot(mpc_planner_2.fsteps[k,3*i+1] , mpc_planner_2.fsteps[k,3*i+2] , "bo" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none'  )
            if i == 1 :
                plt.plot(mpc_planner_2.fsteps[k,3*i+1] , mpc_planner_2.fsteps[k,3*i+2] , "ro" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none' )
            if i == 2 :
                plt.plot(mpc_planner_2.fsteps[k,3*i+1] , mpc_planner_2.fsteps[k,3*i+2] , "ko" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none')
            if i == 3 :
                plt.plot(mpc_planner_2.fsteps[k,3*i+1] , mpc_planner_2.fsteps[k,3*i+2] , "go" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none')

        
plt.show(block=True)



