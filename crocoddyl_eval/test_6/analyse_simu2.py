
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

iteration = 1
dt_mpc = 0.02  # Time step of the MPC
k_mpc = int(dt_mpc / dt)  # dt is dt_tsid, defined in the TSID controller script
n_periods = 1  # Number of periods in the prediction horizon
T_gait = 0.44  # Duration of one gait period

###################
# Parameters
###################

# Perturbation
X_perturb = [0.0,0.0] # [X,Y]
V_perturb = [0.2,0. ]

# Cost weights
# Augmented state weights : 
dt_weight_bound = 0.          # (dt_min - dt)^+ ; (dt-dt_max)^+ for augmented
heuristicWeights = np.full(8,0.0)
lastPositionWeights = np.zeros(8)
forceWeights = np.array(12*[0.01])     # ||u||^2
relative_forces = True                 # ||fz-mg/nb_contact||^2

# Step DT Weights : 
dt_weight_cmd = 10000. # Weight on ||U-dt_ref|| --> Fix dt value
dt_ref = 0.02
dt_weight_bound_cmd = 100000.    # (dt_min - dt)^+ ; (dt-dt_max)^+

Dt_stop1 = [False,False,False]  
dt_ref1 = [0.02 , 0.02 , 0.02]

Dt_stop2 = [False,False,False]  
dt_ref2 = [0.011 , 0.011 , 0.011]

# Step feet Weights : 
vlim = 2.5
speed_weight1 = 0.
speed_weight2 = 0.
stepWeights1 = np.full(4,0.1 )
stepWeights2 = np.full(4,0.0 )

# terminal node
term_factor = 2

# Initialisation
#dt_init = 0.015
dt_init = 0.05
max_iter = 500

#################################################################################################################


#################################
# Update intern MPC parameters
#################################

# MPC optim period 1
mpc_planner_time = MPC_crocoddyl_planner_time(dt = dt_mpc , T_mpc = T_gait, n_periods = n_periods , min_fz = 1)

# Augmented state weights 1 : 
mpc_planner_time.dt_weight_bound = dt_weight_bound
mpc_planner_time.vlim = vlim
mpc_planner_time.heuristicWeights = heuristicWeights
mpc_planner_time.lastPositionWeights = lastPositionWeights
mpc_planner_time.forceWeights = forceWeights
mpc_planner_time.relative_forces = relative_forces

# Step DT Weights 1 : 
mpc_planner_time.dt_weight_cmd = dt_weight_cmd
mpc_planner_time.dt_ref = dt_ref
mpc_planner_time.dt_weight_bound_cmd = dt_weight_bound_cmd

# Step feet Weights 1 : 
mpc_planner_time.speed_weight = speed_weight1
mpc_planner_time.stepWeights = stepWeights1
###################
# MPC optim period 2
mpc_planner_time_2 = MPC_crocoddyl_planner_time(dt = dt_mpc , T_mpc = T_gait , n_periods = n_periods , min_fz = 1)

# Augmented state weights : 
mpc_planner_time_2.dt_weight_bound = dt_weight_bound
mpc_planner_time_2.vlim = vlim
mpc_planner_time_2.heuristicWeights = heuristicWeights
mpc_planner_time_2.lastPositionWeights = lastPositionWeights
mpc_planner_time_2.forceWeights = forceWeights
mpc_planner_time_2.relative_forces = relative_forces

# Step DT Weights : 
mpc_planner_time_2.dt_weight_cmd = dt_weight_cmd
mpc_planner_time_2.dt_ref = dt_ref
mpc_planner_time_2.dt_weight_bound_cmd = dt_weight_bound_cmd

# Step feet Weights : 
mpc_planner_time_2.speed_weight = speed_weight2
mpc_planner_time_2.stepWeights = stepWeights2


#################################
# Create Gait matrix
#################################

gait = np.zeros((20,5))
gait[0,:] = np.array([2,1,0,0,1])
gait[1,:] = np.array([1,1,1,1,1])
gait[2,:] = np.array([10,0,1,1,0])
gait[3,:] = np.array([1,1,1,1,1])
gait[4,:] = np.array([10,1,0,0,1])
gait[5,:] = np.array([1,1,1,1,1])
print(gait)

n_nodes = np.int(T_gait/dt_mpc*n_periods +  1 + gait[0,0])

# Update of the list model
xref = np.zeros((12,n_nodes + 1))
xref[2,:] = 0.2027

l_feet = l_feet_[:,:,0] 

# On swing phase before --> initialised below shoulder
p0 = np.zeros(8)
p_shoulder = [ 0.1946,0.15005, 0.1946,-0.15005, -0.1946,   0.15005 ,-0.1946,  -0.15005]
p0 = np.repeat(np.array([1,1,1,1])-gait[0,1:],2)*p_shoulder   
# On the ground before -->  initialised with the current feet position
p0 +=  np.repeat(gait[0,1:],2)*l_feet[0:2,:].reshape(8, order = 'F')


####################################################
# DDP PERIOD OPTIMISATION                          #
####################################################

def run_MPC_optim_period(mpc ,  Dt_stop , dt_ref , nb_mpc) : 

    # Perturbation 
    xref[6,0] = V_perturb[0]
    xref[7,0] = V_perturb[1]
    xref[0,0] = X_perturb[0]
    xref[1,0] = X_perturb[1]

    j = 0
    k_cum = 0
    ListAction = []

    # WARM START
    l_fsteps = np.zeros((3,4))  
    x1 = np.zeros(21)
    x1[2] = 0.2027
    x1[-1] = dt_init
    u1 = np.array([0.1,0.1,8,0.1,0.1,8,0.1,0.1,8,0.1,0.1,8])
    x_init = []
    u_init = []
    # Iterate over all phases of the gait
    # The first column of xref correspond to the current state 
    while (gait[j, 0] != 0):
        for i in range(k_cum, k_cum+np.int(gait[j, 0])):

            if np.sum(gait[j, 1:]) == 2 and i == 0  :  
                modelTime = quadruped_walkgen.ActionModelQuadrupedTime()
                modelTime.updateModel(np.reshape(l_fsteps, (3, 4), order='F') , xref[:, i]  , gait[j, 1:]) 
                                
                # Update intern parameters
                mpc.update_model_step_time(modelTime , True)
                modelTime.dt_weight_cmd = mpc.dt_weight_cmd*Dt_stop[0]
                modelTime.dt_ref = dt_ref[0]
                ListAction.append(modelTime)   
                x_init.append(x1)
                u_init.append(np.array([dt_init]))

            

            model = quadruped_walkgen.ActionModelQuadrupedAugmentedTime()
            mpc.update_model_augmented(model ,True)

            model.updateModel(np.reshape(l_fsteps, (3, 4), order='F') , xref[:, i+1]  , gait[j, 1:])
            # Update intern parameters
            ListAction.append(model)

            x_init.append(x1)
            u_init.append(np.repeat(gait[j,1:] , 3)*u1)
        
        if np.sum(gait[j+1, 1:]) == 4 : # No optimisation on the first line     
        
            model = quadruped_walkgen.ActionModelQuadrupedStepTime()
            mpc.update_model_step_feet(model , True)

            if nb_mpc == 1 :
                model.speed_weight = speed_weight1
            else : 
                model.speed_weight = 0.
        

            model.updateModel(np.reshape(l_fsteps, (3, 4), order='F') , xref[:, i+1]  ,  gait[j+1, 1:] - gait[j, 1:])
            model.nb_nodes = gait[j,0]
            # Update intern parameters
            ListAction.append(model)
            x_init.append(x1)
            u_init.append(np.zeros(4))
                
        if np.sum(gait[j+1, 1:]) == 2 and j >= 1 :    

            modelTime = quadruped_walkgen.ActionModelQuadrupedTime()
            
                
            modelTime.updateModel(np.reshape(l_fsteps, (3, 4), order='F') , xref[:, i+1]  , gait[j, 1:]) 
            
            # Update intern parameters
            mpc.update_model_step_time(modelTime , True)
            if  j== 1 :
                modelTime.dt_weight_cmd = mpc.dt_weight_cmd*Dt_stop[1]
                modelTime.dt_ref = dt_ref[1]
            if  j== 3 :
                modelTime.dt_weight_cmd = mpc.dt_weight_cmd*Dt_stop[2]
                modelTime.dt_ref = dt_ref[2]
            ListAction.append(modelTime)   
            x_init.append(x1)
            u_init.append(np.array([dt_init]))

        k_cum += np.int(gait[j, 0])
        j += 1


    # Model parameters of terminal node  
    terminalModel = quadruped_walkgen.ActionModelQuadrupedAugmentedTime()
    terminalModel.updateModel(np.reshape(l_fsteps, (3, 4), order='F') , xref[:, -1]  , gait[j-1, 1:]) 
    mpc.update_model_augmented(terminalModel , True)
    x_init.append(np.zeros(21))
    # Weights vectors of terminal node
    terminalModel.forceWeights = np.zeros(12)
    terminalModel.frictionWeights = 0.
    terminalModel.heuristicWeights = np.full(8,0.0)
    terminalModel.lastPositionWeights =  np.full(8,0.0)
    terminalModel.stateWeights = term_factor*terminalModel.stateWeights 


    # Shooting problem
    problem = crocoddyl.ShootingProblem(np.zeros(21),  ListAction, terminalModel)
    problem.x0 = np.concatenate([xref[:,0] , p0 , [dt_init]   ])
    ddp = crocoddyl.SolverDDP(problem)
    ddp.setCallbacks([crocoddyl.CallbackVerbose() ])
    ddp.solve(x_init,u_init,max_iter)

    return ddp


################
# Get results  #
################

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

###############################################################################################################################

# Run optim 1 & 2 
ddp1 = run_MPC_optim_period(mpc_planner_time , Dt_stop1 , dt_ref1 , 1 )
ddp2 = run_MPC_optim_period(mpc_planner_time_2 , Dt_stop2 ,  dt_ref2 , 2)
#ddp1.solve(ddp2.xs,ddp2.us,10000,isFeasible=True)
#get results
Xs_1 , Us_1 , lt_state_1 , lt_force_1 , Cost_1 , x_dt_change_1 , x_foot_change_1 , results_dt_1 , fsteps_1 = get_results(ddp1)
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


plt.suptitle("cost_1 : " + str(ddp1.cost) + "  ; iter_1 : " + str(ddp1.iter) +"    /   cost_2 : " + str(ddp2.cost) + "  ; iter_2 : " + str(ddp2.iter) )

plt.legend()


#############
#  Plot     #
#############

# Predicted evolution of state variables

l_str2 = ["X_ddp", "Y_ddp", "Z_ddp", "Roll_ddp", "Pitch_ddp", "Yaw_ddp", "Vx_ddp", "Vy_ddp", "Vz_ddp", "VRoll_ddp", "VPitch_ddp", "VYaw_ddp"]

index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
plt.figure()

plt.suptitle("State prediction : comparison between control cycle with dt = 0.02 fixed and dt optimisation")
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
plt.suptitle("Contact forces : comparison between control cycle with dt = 0.02 fixed and dt optimisation")
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
        plt.plot(fsteps_1[0,3*i+1] , fsteps_1[0,3*i+2] , "ko" , markerSize = 7   )
    else : 
        plt.plot(p0[2*i] , p0[2*i+1] , "kx" , markerSize = 8   )

# Position of the center of mass
for elt in Xs_1 : 
    plt.plot(Xs_1[0,:] ,Xs_1[1,:] , "gx")


for i in range(4) : 
    for k in range(1,index) : 
        if fsteps_1[k,3*i+1] != 0. and gait[k,i+1] !=0 and gait[k-1,i+1] == 0  : 
            if i == 0 :
                plt.plot(fsteps_1[k,3*i+1] , fsteps_1[k,3*i+2] , "bo" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none'  )
            if i == 1 :
                plt.plot(fsteps_1[k,3*i+1] , fsteps_1[k,3*i+2] , "ro" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none' )
            if i == 2 :
                plt.plot(fsteps_1[k,3*i+1] , fsteps_1[k,3*i+2] , "ko" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none')
            if i == 3 :
                plt.plot(fsteps_1[k,3*i+1] , fsteps_1[k,3*i+2] , "go" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none')


plt.figure()
for i in range(4) : 
    if gait[0,i+1] == 1 : 
        plt.plot(fsteps_2[0,3*i+1] , fsteps_2[0,3*i+2] , "ko" , markerSize = 7   )
    else : 
        plt.plot(p0[2*i] , p0[2*i+1] , "kx" , markerSize = 8   )

# Position of the center of mass

plt.suptitle("DDP OPTIM 2")
for elt in Xs_2 : 
    plt.plot(Xs_2[0,:] ,Xs_2[1,:] , "gx")


for i in range(4) : 
    for k in range(1,index) : 
        if fsteps_2[k,3*i+1] != 0. and gait[k,i+1] !=0 and gait[k-1,i+1] == 0  : 
            if i == 0 :
                plt.plot(fsteps_2[k,3*i+1] , fsteps_2[k,3*i+2] , "bo" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none'  )
            if i == 1 :
                plt.plot(fsteps_2[k,3*i+1] , fsteps_2[k,3*i+2] , "ro" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none' )
            if i == 2 :
                plt.plot(fsteps_2[k,3*i+1] , fsteps_2[k,3*i+2] , "ko" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none')
            if i == 3 :
                plt.plot(fsteps_2[k,3*i+1] , fsteps_2[k,3*i+2] , "go" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none')


print("\n")
print("Cost : [0,:] --> State Cost")
print("Cost : [1,:] --> Force Cost")
print("Cost : [2,:] --> Friction Cost")
print("Cost : [3,:] --> Delta foot Cost")
print("Cost : [4,:] --> Speed Cost")
print("Cost : [5,:] --> dtCmdBound")
print("Cost : [6,:] --> dtCmdRef")

print("\n")
print("dt_min : " + str(mpc_planner_time.dt_min))
print("dt_max : " + str(mpc_planner_time.dt_max))
print("\n")
print("Dt 1 :" + str(results_dt_1))
print("Dt 2 :" + str(results_dt_2))
# plt.show(block=True)



