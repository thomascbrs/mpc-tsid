
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
from crocoddyl_class.MPC_crocoddyl_planner import *
import Interface
####################
# Recovery of Data
####################

folder_name = "log_eval/"
pathIn = "crocoddyl_eval/test_6/"
# oC = np.load(pathIn + folder_name + "oC.npy" , allow_pickle=True )
o_feet_ = np.load(pathIn + folder_name + "o_feet_.npy" , allow_pickle=True ) # global position of the feet
o_feet_heur = np.load(pathIn + folder_name + "o_feet_heur.npy" , allow_pickle=True )
gait_ = np.load(pathIn + folder_name + "gait_.npy" , allow_pickle=True )
ddp_xs = np.load(pathIn + folder_name + "pred_trajectories.npy" , allow_pickle=True )
ddp_us = np.load(pathIn + folder_name + "pred_forces.npy" , allow_pickle=True )
l_feet_ = np.load(pathIn + folder_name + "l_feet_.npy" , allow_pickle=True ) # Local position of the feet
xref = np.load(pathIn + folder_name + "xref.npy" , allow_pickle=True ) 

lfeet_pos =  np.load(pathIn + folder_name + "lfeet_pos.npy" , allow_pickle=True ) 
lfeet_vel =  np.load(pathIn + folder_name + "lfeet_vel.npy" , allow_pickle=True ) 
lfeet_acc =  np.load(pathIn + folder_name + "lfeet_acc.npy" , allow_pickle=True ) 
####################
# Iteration 
####################

iteration = 15
dt_mpc = 0.01  # Time step of the MPC
k_mpc = int(dt_mpc / dt)  # dt is dt_tsid, defined in the TSID controller script
n_periods = 1  # Number of periods in the prediction horizon
T_gait = 0.28  # Duration of one gait period

# Create Interface object
interface = Interface.Interface()

# MPC optim period 1
mpc_planner_time = MPC_crocoddyl_planner(dt = dt_mpc , T_mpc = fstep_planner.T_gait , n_periods = n_periods)
mpc_planner_time_2 = MPC_crocoddyl_planner_time(dt = dt_mpc , T_mpc = T_gait, n_periods = n_periods , min_fz = 1)
# mpc_planner_time.term_factor = 5
# mpc_planner_time.vlim = 1.5


mpc_planner_time_2.speed_weight2 = 0.1
mpc_planner_time_2.speed_weight_first = 2
mpc_planner_time_2.stepWeights = np.array([0.05,0.1,0.05,0.1])   
mpc_planner_time_2.stepWeights2 = np.array([0.05,0.1,0.05,0.1]) 
mpc_planner_time_2.dt_weight_bound_cmd = 100000. 
mpc_planner_time_2.vlim = 1.

mpc_planner_time.speed_weight2 = 0
mpc_planner_time_2.speed_weight_first = 1
# # Weight on the shoulder term : 
# mpc_planner_time.shoulderWeights = 0.1
# mpc_planner_time.shoulder_hlim = 0.21 

# # Cost weights
# # Augmented state weights : 
# mpc_planner_time.dt_weight_bound = 0.  
# mpc_planner_time.forceWeights = np.array(12*[0.01])     # ||u||^2
# mpc_planner_time.relative_forces = True                 # ||fz-mg/nb_contact||^2

# # Step DT Weights : 
# mpc_planner_time.dt_weight_cmd = 10000. # Weight on ||U-dt_ref|| --> Fix dt value
# mpc_planner_time.dt_ref = 0.02
# mpc_planner_time.dt_weight_bound_cmd = 10000.    # (dt_min - dt)^+ ; (dt-dt_max)^+
# mpc_planner_time.stepWeights = np.array([0.1,0.3,0.1,0.3])   
# mpc_planner_time.stepWeights2 = np.array([0.1,0.2,0.1,0.2]) 


# mpc_planner_time.solve(k, fstep_planner.xref , interface.l_feet , interface.oMl)
for i in range(iteration-1) : 
    # if i > 0:            
    #     mpc_planner_time.roll()         
    # else : 
    #     # Create gait matrix
    #     mpc_planner_time.create_walking_trot()
    interface.l_feet = lfeet_pos[:,:,i*k_mpc]
    interface.lv_feet = lfeet_vel[:,:,i*k_mpc]
    interface.la_feet = lfeet_acc[:,:,i*k_mpc]
    if i == 100+(iteration - 2 ) :
        xref2 = xref[:,:,i]
        xref2[6,1:] = 0.6
        xref2[7,1:] = 0.6
    else :
        xref2 = xref[:,:,i]
    mpc_planner_time.solve( i , xref2 ,  interface)
    mpc_planner_time_2.solve(i , xref2 , interface )
    # print(i+1)
    # print(xref[:,0,i])

    # for elt in mpc_planner_time.ddp.problem.runningModels : 
 
    #     print(elt.__class__.__name__)

p0 = np.zeros(8)
for k in range(4) : 
    p0[2*k] = l_feet_[:,:,i][0,k]
    p0[2*k+1] = l_feet_[:,:,i][1,k]

p0[2] = 0.19
p0[3] = 0.15

# xref[6:9,0,i] = np.array([1.5,1.5,0.0])

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
            # x_foot_change.append(lt_state[index-gap])
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

# mpc_planner_time.updateProblem( iteration , xref[:,:,iteration] , l_feet_[:,:,iteration] )
# # Solve problem
# mpc_planner_time.ddp.solve(mpc_planner_time.x_init,mpc_planner_time.u_init, 50)  


ddp1 = mpc_planner_time.ddp
gait = mpc_planner_time.gait
Xs_1 , Us_1 , lt_state_1 , lt_force_1 , Cost_1 , x_dt_change_1 , x_foot_change_1 , results_dt_1 , fsteps_1 = get_results(ddp1)

# for i in range(len(mpc_planner_time.ddp.problem.runningModels)) : 
#     for i in range(len(mpc_planner_time.ddp.problem.runningModels)) : 
#             if mpc_planner_time.ddp.problem.runningModels[i].nu == 4 :                 
#                 mpc_planner_time.ddp.problem.runningModels[i].speed_weight = mpc_planner_time.speed_weight2
#                 mpc_planner_time.ddp.problem.runningModels[i].stepWeights = mpc_planner_time.stepWeights
#             if mpc_planner_time.ddp.problem.runningModels[i].nu == 1 :  
#                 mpc_planner_time.ddp.problem.runningModels[i].dt_weight_cmd = 0.  

# mpc_planner_time.ddp.solve(mpc_planner_time.ddp.xs,mpc_planner_time.ddp.us,100, isFeasible=True) 

ddp2 = mpc_planner_time_2.ddp
Xs_2 , Us_2 , lt_state_2 , lt_force_2 , Cost_2 , x_dt_change_2 , x_foot_change_2 , results_dt_2 , fsteps_2 = get_results(ddp2)

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
    plt.plot(lt_state_2 , Cost_2[i,:] , color[i] + "x--", label = legend[i] + "_2")


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
    pl2, = plt.plot(lt_state_2, Xs_2[i,:], linewidth=2, marker='x' , label = "2")
    first_legend = plt.legend(handles=[pl1,pl2])
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
    pl2, = plt.plot(lt_force_2,Us_2[i,:], linewidth=2, marker='x' , label = "2")

    first_legend = plt.legend(handles=[pl1,pl2])
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

plt.figure()
for i in range(4) : 
    if gait[0,i+1] == 1 : 
        pl7, = plt.plot(fsteps_2[0,3*i+1] , fsteps_2[0,3*i+2] , "ko" , markerSize = 7   )
    else : 
        pl8, = plt.plot(p0[2*i] , p0[2*i+1] , "kx" , markerSize = 8   )

# Position of the center of mass

plt.suptitle("DDP OPTIM 2")
for elt in Xs_2 : 
    pl6, = plt.plot(Xs_2[0,:] ,Xs_2[1,:] , "gx-")


for i in range(4) : 
    for k in range(1,index) : 
        if fsteps_2[k,3*i+1] != 0. and gait[k,i+1] !=0 and gait[k-1,i+1] == 0  : 
            if k >= 0 :
                if i == 0 :
                    pl1, = plt.plot(fsteps_2[k,3*i+1] , fsteps_2[k,3*i+2] , "bo" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none'  )
                if i == 1 :
                    pl2, = plt.plot(fsteps_2[k,3*i+1] , fsteps_2[k,3*i+2] , "ro" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none' )
                if i == 2 :
                    pl3, = plt.plot(fsteps_2[k,3*i+1] , fsteps_2[k,3*i+2] , "ko" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none')
                if i == 3 :
                    pl4, = plt.plot(fsteps_2[k,3*i+1] , fsteps_2[k,3*i+2] , "go" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none')
            else : 
                if i == 0 :
                    plt.plot(fsteps_2[k,3*i+1] , fsteps_2[k,3*i+2] , "bo" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none'  )
                if i == 1 :
                    plt.plot(fsteps_2[k,3*i+1] , fsteps_2[k,3*i+2] , "ro" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none' )
                if i == 2 :
                    plt.plot(fsteps_2[k,3*i+1] , fsteps_2[k,3*i+2] , "ko" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none')
                if i == 3 :
                    plt.plot(fsteps_2[k,3*i+1] , fsteps_2[k,3*i+2] , "go" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none')

# Centre of pressure
for j in range(len(ddp2.us)) : 
    if ddp2.us[j].size == 12 : 
        fz = np.array([ddp2.us[j][i] for i in range(len(ddp2.us[4])) if (i+1)%3 == 0 ])
        dx = np.array([ddp2.xs[j][12 +i] for i in range(8) if i%2 == 0])
        dy = np.array([ddp2.xs[j][12 +i] for i in range(8) if i%2 == 1])
        if j == 7 :
            pl5, = plt.plot(np.sum(dx*fz)/np.linalg.norm(fz) , np.sum(dy*fz)/np.linalg.norm(fz) , "mx" , markerSize = int(20/np.sqrt(j)) )
        else : 
            plt.plot(np.sum(dx*fz)/np.sum(fz) , np.sum(dy*fz)/np.sum(fz) , "mx" , markerSize = int(20/np.sqrt(j+1)) )

plt.legend([pl1,pl2,pl3,pl4,pl5,pl6, pl7 , pl8] , ["FL" , "FR" , "HL" , "HR" ,"CoP" , "CoM" , "Initial contact" , "Pos shoulder"])
plt.suptitle("DDP2 : Foot position, cost ||u||^2 : "  )
plt.xlabel("x")
plt.ylabel("y")


print("\n")
print("Cost : [0,:] --> State Cost")
print("Cost : [1,:] --> Force Cost")
print("Cost : [2,:] --> Friction Cost")
print("Cost : [3,:] --> Delta foot Cost")
print("Cost : [4,:] --> Speed Cost")
print("Cost : [5,:] --> dtCmdBound")
print("Cost : [6,:] --> dtCmdRef")

dt_min_l = [ddp1.problem.runningModels[i].dt_min for i in range(len(ddp1.problem.runningModels)) if ddp1.problem.runningModels[i].nu == 1 ]
dt_max_l = [ddp1.problem.runningModels[i].dt_max for i in range(len(ddp1.problem.runningModels)) if ddp1.problem.runningModels[i].nu == 1 ]
print("\n")
print("dt_min : " + str(dt_min_l[0]) + "  dt_max : " + str(dt_max_l[0]))
print("dt_min : " + str(dt_min_l[1]) + "  dt_max : " + str(dt_max_l[1]))
print("dt_min : " + str(dt_min_l[2]) + "  dt_max : " + str(dt_max_l[2]))
print("\n")
print("Dt 1 :" + str(results_dt_1))
print("Dt 2 :" + str(results_dt_2))
