
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
from TSID_Debug_controller_four_legs_fb_vel import controller, dt
import MPC_Wrapper 
import FootstepPlanner
from crocoddyl_class.MPC_crocoddyl_planner_time import *

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

iteration = 15
dt_mpc = 0.01  # Time step of the MPC
k_mpc = int(dt_mpc / dt)  # dt is dt_tsid, defined in the TSID controller script
n_periods = 1  # Number of periods in the prediction horizon
T_gait = 0.28  # Duration of one gait period

# MPC optim period 1
mpc_planner_time = MPC_crocoddyl_planner_time(dt = dt_mpc , T_mpc = T_gait, n_periods = n_periods , min_fz = 1)

# mpc_planner_time.solve(k, fstep_planner.xref , interface.l_feet , interface.oMl)
for i in range(iteration) : 
    mpc_planner_time.solve(i, xref[:,:,i] , l_feet_[:,:,i] )



ddp1 = mpc_planner_time.ddp
gait = mpc_planner_time.gait
p0 = np.zeros(8)
for k in range(4) : 
    p0[2*k] = l_feet_[:,:,i][0,k]
    p0[2*k+1] = l_feet_[:,:,i][1,k]

#################################################################################################################
def get_results(ddp) :

    # Forces
    Us = ddp.us
    Liste = [x for x in Us if (x.size != 4 and x.size != 1) ]
    Us =  np.array(Liste)[:,:].transpose()

    # States
    Xs = [ddp.xs[i] for i in range(len(ddp.us)) if (ddp.us[i].size != 4 and ddp.us[i].size != 1) ]
    Xs.append(ddp.xs[-1]) #terminal node
    Xs = np.array(Xs).transpose()

    # Dt optimised
    reults_dt = [ddp.xs[i+1][-1] for i in range(len(ddp.us)) if ddp.us[i].size == 1 ]

    fsteps = np.zeros((20,13))
    fsteps[0,0] = gait[0,0]

    ListDt = []

    # Iterate over all phases of the gait
    k_cum = 0
    j = 0
    while (gait[j, 0] != 0):
        fsteps[j ,1: ] = np.repeat(gait[j,1:] , 3)*np.concatenate([Xs[12:14 , k_cum ],[0.],Xs[14:16 , k_cum ],[0.],
                                                                            Xs[16:18 , k_cum ],[0.],Xs[18:20 , k_cum ],[0.]])  
    
        ListDt.append(Xs[20 , k_cum ])
        k_cum += np.int(gait[j, 0])
        j += 1      

    lt_state =  [0.]
    for i,elt in enumerate(ListDt) :    
        for j in range(int(gait[i,0])) : 
            lt_state.append(lt_state[-1] + elt)

    lt_force = lt_state.copy()
    lt_force.pop(-1) # no terminal point

    # get cost
    gap = 0

    stateCost = np.zeros(len(lt_state) )
    forceCost = np.zeros(len(lt_state))
    frictionCost = np.zeros(len(lt_state) )
    deltaFoot = np.zeros(len(lt_state))
    speedCost = np.zeros(len(lt_state))
    dtCmdBound = np.zeros(len(lt_state))
    dtCmdRef = np.zeros(len(lt_state))

    x_dt_change = []
    x_foot_change = []
    print("-------------------------------")
    for index,elt in enumerate(ddp.problem.runningModels ):
        if elt.__class__.__name__ == "ActionModelQuadrupedAugmentedTime" :  
            stateCost[index-gap] = elt.Cost[0]
            forceCost[index-gap] = elt.Cost[2]
            frictionCost[index-gap] = elt.Cost[5]
        elif elt.__class__.__name__ == "ActionModelQuadrupedTime" :     
            x_dt_change.append(lt_state[index-gap])
            dtCmdBound[index-gap] = elt.Cost[3]
            dtCmdRef[index-gap] = elt.Cost[2]
            gap += 1
            
        else : 
            x_foot_change.append(lt_state[index-gap])
            speedCost[index-gap] = elt.Cost[3]
            deltaFoot[index-gap] = elt.Cost[2]
            gap += 1
    
    stateCost[- 1] = ddp.problem.terminalModel.Cost[0]

    # get Matrix of cost
    Cost = np.concatenate(([stateCost],[forceCost],[frictionCost],[deltaFoot],[speedCost],[dtCmdBound],[dtCmdRef]) , axis = 0)


    return Xs , Us , lt_state , lt_force , Cost , x_dt_change , x_foot_change , reults_dt , fsteps

#################################################

#################################
# Update intern MPC parameters
#################################


Xs_1 , Us_1 , lt_state_1 , lt_force_1 , Cost_1 , x_dt_change_1 , x_foot_change_1 , results_dt_1 , fsteps_1 = get_results(ddp1)


#############
# Plot cost
#############

plt.figure()

# for elt in x_foot_change_1 : 
#     pl8 = plt.axvline(x=elt,color='gray',linestyle='--')

# for elt in x_dt_change_1 : 
#     pl9 = plt.axvline(x=elt,color='gray',linestyle=(0, (1, 10)) )

legend = ["stateCost" , "forceCost" , "frictionCost" , "deltaFoot" , "speedCost" , "dtCmdBound" , "dtCmdRef"]
color = ["k" , "b" , "y" , "r" , "g" , "m" , "c"]

for i in range(Cost_1.shape[0]) : 
    plt.plot(lt_state_1 , Cost_1[i,:] , color[i] + "x-" , label = legend[i])
    # plt.plot(lt_state_2 , Cost_2[i,:] , color[i] + "x--", label = legend[i] + "_2")


plt.suptitle("cost_1 : " + str(ddp1.cost) + "  ; iter_1 : " + str(ddp1.iter) )

plt.legend()


#############
#  Plot     #
#############

# Predicted evolution of state variables

l_str2 = ["X_ddp", "Y_ddp", "Z_ddp", "Roll_ddp", "Pitch_ddp", "Yaw_ddp", "Vx_ddp", "Vy_ddp", "Vz_ddp", "VRoll_ddp", "VPitch_ddp", "VYaw_ddp"]

index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
plt.figure()

plt.suptitle("State prediction : comparison between control cycle with delta step = 0.02 --> ddp2 and 0.2 --> ddp1, dt = 0.02")
for i in range(12):
    plt.subplot(3, 4, index[i])
           
    pl1, = plt.plot(lt_state_1, Xs_1[i,:], linewidth=2, marker='x' , label = l_str2[i])
    # pl2, = plt.plot(lt_state_2, Xs_2[i,:], linewidth=2, marker='x' , label = "2")
    first_legend = plt.legend(handles=[pl1])
    ax = plt.gca().add_artist(first_legend)
       

# plt.legend(handles=[pl3, pl4], title='Legend for time optim : ', bbox_to_anchor=(1.05, 1), loc='upper left')


l_str2 = ["FL_X_ddp", "FL_Y_ddp", "FL_Z_ddp", "FR_X_ddp", "FR_Y_ddp", "FR_Z_ddp", "HL_X_ddp", "HL_Y_ddp", "HL_Z_ddp", "HR_X_ddp", "HR_Y_ddp", "HR_Z_ddp"]
index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
plt.figure()
plt.suptitle("Contact forces : comparison between control cycle with delta step = 0.02 -->ddp2 and 0.2 -->ddp1, dt = 0.02")
for i in range(12):
    plt.subplot(3, 4, index[i])    
    # for elt in x_dt_change : 
    #     pl3 = plt.axvline(x=elt,color='gray',linestyle=(0, (1, 10)) , label = "dt change")
    
    # for elt in x_foot_change : 
    #     pl4 = plt.axvline(x=elt,color='gray',linestyle='--' , label = "foot step")
   
   
    pl1, = plt.plot(lt_force_1,Us_1[i,:], linewidth=2, marker='x' , label = l_str2[i] )
    # pl2, = plt.plot(lt_force_2,Us_2[i,:], linewidth=2, marker='x' , label = "2")

    first_legend = plt.legend(handles=[pl1])
    ax = plt.gca().add_artist(first_legend)
     # plt.legend([pl1,pl2] , [l_str2[i] , "ddp_time" ])

# plt.legend(handles=[pl3, pl4], title='Legend for time optim : ', bbox_to_anchor=(1.05, 1), loc='upper left')


# index = next((idx for idx, val in np.ndenumerate(mpc_planner.gait[:, 0]) if val==0.0), 0.0)[0]

# # Feet on the ground at the beginning : k - o
# # inital position for optimisation :   k - x
# for i in range(4) : 
#     if mpc_planner.gait[0,i+1] == 1 : 
#         plt.plot(mpc_planner.fsteps[0,3*i+1] , mpc_planner.fsteps[0,3*i+2] , "ko" , markerSize = 7   )
#     else : 
#         plt.plot(mpc_planner.p0[2*i] , mpc_planner.p0[2*i+1] , "kx" , markerSize = 8   )

# # Position of the center of mass
# for elt in mpc_planner.Xs : 
#     plt.plot(mpc_planner.Xs[0,:] ,mpc_planner.Xs[1,:] , "gx")


# for i in range(4) : 
#     for k in range(1,index) : 
#         if mpc_planner.fsteps[k,3*i+1] != 0. and mpc_planner.gait[k,i+1] !=0 and mpc_planner.gait[k-1,i+1] == 0  : 
#             if i == 0 :
#                 plt.plot(mpc_planner.fsteps[k,3*i+1] , mpc_planner.fsteps[k,3*i+2] , "bo" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none'  )
#             if i == 1 :
#                 plt.plot(mpc_planner.fsteps[k,3*i+1] , mpc_planner.fsteps[k,3*i+2] , "ro" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none' )
#             if i == 2 :
#                 plt.plot(mpc_planner.fsteps[k,3*i+1] , mpc_planner.fsteps[k,3*i+2] , "ko" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none')
#             if i == 3 :
#                 plt.plot(mpc_planner.fsteps[k,3*i+1] , mpc_planner.fsteps[k,3*i+2] , "go" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none')
index = next((idx for idx, val in np.ndenumerate(gait[:, 0]) if val==0.0), 0.0)[0]

plt.figure()
plt.suptitle("DDP OPTIM 1")
for i in range(4) : 
    if gait[0,i+1] == 1 : 
        pl7, = plt.plot(fsteps_1[0,3*i+1] , fsteps_1[0,3*i+2] , "ko" , markerSize = 7   )
    else : 
        pl8, = plt.plot(p0[2*i] , p0[2*i+1] , "kx" , markerSize = 8   )

# Position of the center of mass
for elt in Xs_1 : 
    pl6, = plt.plot(Xs_1[0,:] ,Xs_1[1,:] , "gx-")

# Centre of pressure
for j in range(len(ddp1.us)) : 
    if ddp1.us[j].size == 12 : 
        fz = np.array([ddp1.us[j][i] for i in range(len(ddp1.us[4])) if (i+1)%3 == 0 ])
        dx = np.array([ddp1.xs[j][12 +i] for i in range(8) if i%2 == 0])
        dy = np.array([ddp1.xs[j][12 +i] for i in range(8) if i%2 == 1])
        if j >= 7 :
            pl5, = plt.plot(np.sum(dx*fz)/np.linalg.norm(fz) , np.sum(dy*fz)/np.linalg.norm(fz) , "mx" , markerSize = int(20/np.sqrt(j)) )
        else : 
            plt.plot(np.sum(dx*fz)/np.sum(fz) , np.sum(dy*fz)/np.sum(fz) , "mx" , markerSize = int(20/np.sqrt(j+1)) )


for i in range(4) : 
    for k in range(1,index) : 
        if fsteps_1[k,3*i+1] != 0. and gait[k,i+1] !=0 and gait[k-1,i+1] == 0  : 
            if k >= 0 :
                if i == 0 :
                    pl1, = plt.plot(fsteps_1[k,3*i+1] , fsteps_1[k,3*i+2] , "bo" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none'  )
                if i == 1 :
                    pl2, = plt.plot(fsteps_1[k,3*i+1] , fsteps_1[k,3*i+2] , "ro" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none' )
                if i == 2 :
                    pl3, = plt.plot(fsteps_1[k,3*i+1] , fsteps_1[k,3*i+2] , "ko" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none')
                if i == 3 :
                    pl4, = plt.plot(fsteps_1[k,3*i+1] , fsteps_1[k,3*i+2] , "go" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none')
            else : 
                if i == 0 :
                    plt.plot(fsteps_1[k,3*i+1] , fsteps_1[k,3*i+2] , "bo" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none'  )
                if i == 1 :
                    plt.plot(fsteps_1[k,3*i+1] , fsteps_1[k,3*i+2] , "ro" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none' )
                if i == 2 :
                    plt.plot(fsteps_1[k,3*i+1] , fsteps_1[k,3*i+2] , "ko" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none')
                if i == 3 :
                    plt.plot(fsteps_1[k,3*i+1] , fsteps_1[k,3*i+2] , "go" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none')

plt.legend([pl1,pl2,pl3,pl4,pl5,pl6, pl7 , pl8] , ["FL" , "FR" , "HL" , "HR" ,"CoP" , "CoM" , "Initial contact" , "Pos shoulder"])
plt.suptitle("DDP1 : Foot position, cost ||u||^2 : ")
plt.xlabel("x")
plt.ylabel("y")
###################################################

print("\n")
print("Cost : [0,:] --> State Cost")
print("Cost : [1,:] --> Force Cost")
print("Cost : [2,:] --> Friction Cost")
print("Cost : [3,:] --> Delta foot Cost")
print("Cost : [4,:] --> Speed Cost")
print("Cost : [5,:] --> dtCmdBound")
print("Cost : [6,:] --> dtCmdRef")

print("\n")
for i in range(len(ddp1.us)) : 
    if ddp1.problem.runningModels[i].nu == 1 : 
        print("dt_min : " + str(ddp1.problem.runningModels[i].dt_min) + "    /   dt_max : " + str(ddp1.problem.runningModels[i].dt_max))
# print("dt_min : " + str(mpc_planner_time.dt_min))
print("\n")
print("Dt 1 :" + str(results_dt_1))