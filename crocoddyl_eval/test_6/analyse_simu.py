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

iteration = 1
dt_mpc = 0.02  # Time step of the MPC
k_mpc = int(dt_mpc / dt)  # dt is dt_tsid, defined in the TSID controller script
n_periods = 1  # Number of periods in the prediction horizon
T_gait = 0.44  # Duration of one gait period

###################
# Parameters 
###################
X_perturb = [0.0,0.0] # [X,Y]
# V_perturb = [0.0,0.26]  # V = [Vx , Vy]
V_perturb = [0.0,0.0]
# Plot cost 1D : 
cost_1D = True
Dt_list = [0.02,0.02,0.02]
Dt_free = [True, False,False]

# Plot cost 2D :
cost_2D = False
Dt_list = [0.02,0.02,0.02]
Dt_free2 = [True, True,False]

# Plot cost 3D : 
cost_3D = False

# MPC optim period
mpc_planner_time = MPC_crocoddyl_planner_time(dt = dt_mpc , T_mpc = 0.32 , n_periods = n_periods , min_fz = 1)

# Perturbation 
Vx = V_perturb[0]
Vy = V_perturb[1]
x = X_perturb[0]
y = X_perturb[1]

# WAugmented state weights : 
mpc_planner_time.dt_weight_bound = 0.
mpc_planner_time.vlim = 1.5
mpc_planner_time.heuristicWeights = np.full(8,0.0)
mpc_planner_time.lastPositionWeights = np.zeros(8)

# Step DT Weights : 
mpc_planner_time.dt_weight_cmd = 0. # Weight on ||U-dt_ref||
mpc_planner_time.dt_weight_bound_cmd = 100000.

# Step feet Weights : 
mpc_planner_time.speed_weight = 100.
mpc_planner_time.stepWeights = np.full(4,0.8)

dt_init = 0.01

term_factor = 10
Dt_stop = [False,False,False]
cmd_weight = mpc_planner_time.dt_weight_cmd 

######################################
#  DDP FEET OPTIMISATION             #
######################################

Relaunch_DDP = True
mpc_planner = MPC_crocoddyl_planner(dt = dt_mpc , T_mpc = T_gait , n_periods = n_periods)


# mpc_planner.stepWeights = np.full(4,0.0)   
# mpc_planner.heuristicWeights = np.full(8,0.0)
# mpc_planner.frictionWeights = 0.
# mpc_planner.shoulderWeights = 0.
# mpc_planner.lastPositionWeights = np.full(8,0.)


mpc_planner.gait = np.zeros((20,5))
mpc_planner.gait[0,:] = np.array([2,1,0,0,1])
mpc_planner.gait[1,:] = np.array([1,1,1,1,1])
mpc_planner.gait[2,:] = np.array([10,0,1,1,0])
mpc_planner.gait[3,:] = np.array([1,1,1,1,1])
mpc_planner.gait[4,:] = np.array([10,1,0,0,1])
mpc_planner.gait[5,:] = np.array([1,1,1,1,1])
gait = mpc_planner.gait
print(gait)

n_nodes = np.int(T_gait/dt_mpc*n_periods +  1 + gait[0,0])

# Update of the list model
xref = np.zeros((12,n_nodes + 1))
xref[2,:] = 0.2027

l_feet = l_feet_[:,:,0] 

# On swing phase before --> initialised below shoulder
p0 = np.zeros(8)
p0 = np.repeat(np.array([1,1,1,1])-mpc_planner.gait[0,1:],2)*mpc_planner.p0  
# On the ground before -->  initialised with the current feet position
p0 +=  np.repeat(mpc_planner.gait[0,1:],2)*l_feet[0:2,:].reshape(8, order = 'F')


###################################################################################
####        MPC FOOT STEP, MANUALLY TUNE DT
###################################################################################

def run_planner(Dt_list , V_perturb) : 

    cost = 0

    mpc_planner.ListAction = []
    mpc_planner.problem = None 
    mpc_planner.ddp = None

    mpc_planner.stateWeights[1] = 0.3
    # Gait matrix
    
    mpc_planner.Xs = np.zeros((20,int(T_gait/dt_mpc)*n_periods + int(np.sum(gait[:2,0])) ))


    # Creation of the list model
    mpc_planner.create_List_model()
    
    xref[0,0] = X_perturb[0]
    xref[1,0] = X_perturb[1]
    xref[6,0] = V_perturb[0]
    xref[7,0] = V_perturb[1]

    # mpc_planner.l_fsteps is used to gives the position of the feet computed by the previous gait cycle
    # Here the cost is set to 0, thus mpc_planner.l_fsteps = np.zeros(4,3)

    # On swing phase before --> initialised below shoulder
    p0 = np.zeros(8)
    p0 = np.repeat(np.array([1,1,1,1])-mpc_planner.gait[0,1:],2)*mpc_planner.p0  
    # On the ground before -->  initialised with the current feet position
    p0 +=  np.repeat(mpc_planner.gait[0,1:],2)*l_feet[0:2,:].reshape(8, order = 'F')

    j = 0
    k_cum = 0
    L = []
    # Iterate over all phases of the gait
    # The first column of xref correspond to the current state 
    # Gap introduced to take into account the Step model (more nodes than gait phases )
    x_init = []
    u_init = []
    gap = 0
    while (mpc_planner.gait[j, 0] != 0):    
        for i in range(k_cum, k_cum+np.int(mpc_planner.gait[j, 0])):

            if mpc_planner.ListAction[i].__class__.__name__ == "ActionModelQuadrupedStep" :
                x_init.append(np.zeros(20))
                u_init.append(np.zeros(4))

                if j == 0 or j == 1 :
                    mpc_planner.ListAction[i].dt = Dt_list[0]
                    mpc_planner.ListAction[i+1].dt = Dt_list[0]
                
                if j == 2 or j == 3 :
                    mpc_planner.ListAction[i].dt = Dt_list[1]
                    mpc_planner.ListAction[i+1].dt = Dt_list[1]
                
                if j == 4 or j == 5 :
                    mpc_planner.ListAction[i].dt = Dt_list[2]
                    mpc_planner.ListAction[i+1].dt = Dt_list[2]


                mpc_planner.ListAction[i].updateModel(np.reshape(mpc_planner.l_fsteps, (3, 4), order='F') , xref[:, i+gap]  , mpc_planner.gait[j, 1:] - mpc_planner.gait[j-1, 1:])
                mpc_planner.ListAction[i+1].updateModel(np.reshape(mpc_planner.l_fsteps, (3, 4), order='F') , xref[:, i+gap]  , mpc_planner.gait[j, 1:])
                x_init.append(np.zeros(20))
                u_init.append(np.zeros(12))
                k_cum +=  1
                gap -= 1
                # self.ListAction[i+1].shoulderWeights = 2*np.array(4*[0.25,0.3])


                
            else : 
                if j == 0 or j == 1 :
                    mpc_planner.ListAction[i].dt = Dt_list[0]
                if j == 2 or j == 3 :
                    mpc_planner.ListAction[i].dt = Dt_list[1]
                if j == 4 or j == 5 :
                    mpc_planner.ListAction[i].dt = Dt_list[2]

                mpc_planner.ListAction[i].updateModel(np.reshape(mpc_planner.l_fsteps, (3, 4), order='F') , xref[:, i+gap]  , mpc_planner.gait[j, 1:])                    
                x_init.append(np.zeros(20))
                u_init.append(np.zeros(12))

        k_cum += np.int(mpc_planner.gait[j, 0])
        j += 1 

    # Update model of the terminal model
    mpc_planner.terminalModel.updateModel(np.reshape(mpc_planner.fsteps[j-1, 1:], (3, 4), order='F') , xref[:,-1] , mpc_planner.gait[j-1, 1:])
    x_init.append(np.zeros(20))

    # Shooting problem
    mpc_planner.problem = crocoddyl.ShootingProblem(np.zeros(20), mpc_planner.ListAction, mpc_planner.terminalModel)
    mpc_planner.problem.x0 = np.concatenate([xref[:,0] , p0   ])

    # DDP Solver
    mpc_planner.ddp = crocoddyl.SolverDDP(mpc_planner.problem)
    mpc_planner.ddp.solve(x_init,u_init,20)
    mpc_planner.get_fsteps()

    for index, elt in enumerate(mpc_planner.ListAction) : 
        data_ = elt.createData()
        elt.calc(data_ , mpc_planner.ddp.xs[index] ,  mpc_planner.ddp.us[index] )
        cost += data_.cost 


    return cost 

################################################################################################################
#                                     COST PLOT     1D                                                         #
###############################################################################################################


if cost_1D :
    Ldt = np.linspace(0.005 , 0.06 , 30)
    S_cost = np.zeros((30,2))
    S_cost1 = np.zeros((30,2))
    S_cost2 = np.zeros((30,2))
    for index,dt1 in enumerate(Ldt) : 
        if Dt_free[0] : 
            Dt_list[0] = dt1
        elif Dt_free[1] : 
            Dt_list[1] = dt1
        else : 
            Dt_list[2] = dt1
        S_cost[index,:] = np.array([dt1 , run_planner(Dt_list,V_perturb)])

    # Ldt = np.linspace(0.005 , 0.06 , 30)
    # for index,dt1 in enumerate(Ldt) : 
    #     S_cost1[index,:] = np.array([dt1 , run_planner(dt1,0.02,0.3)])

    # Ldt = np.linspace(0.005 , 0.06 , 30)

    # for index,dt1 in enumerate(Ldt) : 
    #     S_cost2[index,:] = np.array([dt1 , run_planner(dt1,0.02,0.5)])

    plt.figure()
    plt.plot(S_cost[:,0] , S_cost[:,1] , "x-")
    # plt.plot(S_cost1[:,0] , S_cost1[:,1] , "x-")
    # plt.plot(S_cost2[:,0] , S_cost2[:,1] , "x-")
    plt.xlabel(" dt  [s]" , fontsize=10)
    plt.ylabel("Cost Function" , fontsize=10)
    plt.title("Cost Function wrt dt" , fontsize=10)


# Vx2 = np.array(mpc_planner.ddp.Vx)
# plt.figure()
# plt.plot(np.arange(Vx2.shape[0]) , Vx2[:,10])
###########
# Cost 2D
###########

if cost_2D :
    plt.figure()
    X = np.linspace(0.005 , 0.06 , 20)
    Y = np.linspace(0.005 , 0.06 , 20)
    S_cost = np.zeros((20,20))
    for i in range(len(X)) : 
        for j in range(len(Y)) : 
            if Dt_free2 == [True,True,False] : 
                Dt_list[0] = X[i]
                Dt_list[1] = Y[j]
            elif Dt_free2 == [True,False,True] : 
                Dt_list[0] = X[i]
                Dt_list[2] = Y[j]
            else : 
                Dt_list[1] = X[i]
                Dt_list[2] = Y[j]    
            S_cost[i,len(Y) - j -1] =  run_planner(Dt_list,V_perturb)

    im = plt.imshow(S_cost ,cmap = plt.cm.Blues , extent=(0.005,0.06,0.005,0.06))

    if Dt_free2 == [True,True,False] : 
        plt.xlabel(" dt1  [s]" , fontsize=10)
        plt.ylabel(" dt2  [s]" , fontsize=10)
    elif Dt_free2 == [True,False,True] : 
        plt.xlabel(" dt1  [s]" , fontsize=10)
        plt.ylabel(" dt3  [s]" , fontsize=10)
    else : 
        plt.xlabel(" dt2  [s]" , fontsize=10)
        plt.ylabel(" dt3  [s]" , fontsize=10) 
 

    results = np.where(S_cost == np.amin(S_cost))
    print("\n")
    print("Optimal dt : ")
    print("dt1 : " , X[results[0]])
    print("dt2 : " , Y[len(Y) -1 -results[1]])
    print("\n")
    print("Period optim : upper and lower bound :")


###########
# Cost 3D
###########

if cost_3D :
    nb = 12
    X = np.linspace(0.005 , 0.06 , nb)
    Y = np.linspace(0.005 , 0.06 , nb)
    Z = np.linspace(0.005 , 0.06 , nb)
    S_cost = np.zeros((nb,nb,nb))
    for i in range(len(X)) : 
        for j in range(len(Y)) : 
            for k in range(len(Y)) :
                Dt_list[0] = X[i]
                Dt_list[1] = Y[j]
                Dt_list[2] = Z[k]
                S_cost[i,j,k] =  run_planner(Dt_list,V_perturb)

    results = np.where(S_cost == np.amin(S_cost))
    print("\n")
    print("Optimal dt : ")
    print("dt1 : " , X[results[0]])
    print("dt2 : " , Y[results[1]])
    print("dt3 : " , Y[results[2]])
    print("\n")
    print("Period optim : upper and lower bound :")




# run_planner(Dt_list,V_perturb)

####################################################
# DDP PERIOD OPTIMISATION                          #
####################################################

Relaunch_TIME = True

mpc_print = MPC_crocoddyl_planner_time(dt = dt_mpc , T_mpc = 0.32 , n_periods = n_periods , min_fz = 1)
print("\n")
# Perturbation 
xref[6,0] = Vx
xref[7,0] = Vy
xref[0,0] = x
xref[1,0] = y

# # WEIGHTS 
# mpc_planner_time.dt_weight_bound = 100000.
# mpc_planner_time.vlim = 1.5
# mpc_planner_time.speed_weight = 100.
# term_factor = 10
# Dt_stop = [False,False,True]
# cmd_weight = 1000


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
            
            modelTime.dt_weight_cmd = mpc_planner_time.dt_weight_cmd*Dt_stop[0]
            # Update intern parameters
            mpc_planner_time.update_model_step_time(modelTime , True)
            ListAction.append(modelTime)   
            x_init.append(x1)
            u_init.append(np.array([dt_init]))

        

        model = quadruped_walkgen.ActionModelQuadrupedAugmentedTime()
        mpc_planner_time.update_model_augmented(model ,True)

        model.updateModel(np.reshape(l_fsteps, (3, 4), order='F') , xref[:, i]  , gait[j, 1:])
        # Update intern parameters
        ListAction.append(model)

        x_init.append(x1)
        u_init.append(np.repeat(gait[j,1:] , 3)*u1)
    
    if np.sum(gait[j+1, 1:]) == 4 : # No optimisation on the first line     
    
        model = quadruped_walkgen.ActionModelQuadrupedStepTime()
        mpc_planner_time.update_model_step_feet(model , True)
    

        model.updateModel(np.reshape(l_fsteps, (3, 4), order='F') , xref[:, i+1]  ,  gait[j+1, 1:] - gait[j, 1:])
        # Update intern parameters
        ListAction.append(model)
        x_init.append(x1)
        u_init.append(np.zeros(4))
            
    if np.sum(gait[j+1, 1:]) == 2 and j >= 1 :    

        modelTime = quadruped_walkgen.ActionModelQuadrupedTime()
        if  j== 1 :
            modelTime.dt_weight_cmd = mpc_planner_time.dt_weight_cmd*Dt_stop[1]
        if  j== 3 :
            modelTime.dt_weight_cmd = mpc_planner_time.dt_weight_cmd*Dt_stop[2]
            
        modelTime.updateModel(np.reshape(l_fsteps, (3, 4), order='F') , xref[:, i+1]  , gait[j, 1:]) 
        # Update intern parameters
        mpc_planner_time.update_model_step_time(modelTime , True)
        ListAction.append(modelTime)   
        x_init.append(x1)
        u_init.append(np.array([dt_init]))

    k_cum += np.int(gait[j, 0])
    j += 1


# Model parameters of terminal node  
terminalModel = quadruped_walkgen.ActionModelQuadrupedAugmentedTime()
terminalModel.updateModel(np.reshape(l_fsteps, (3, 4), order='F') , xref[:, -1]  , gait[j-1, 1:]) 
mpc_planner_time.update_model_augmented(terminalModel , True)
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
ddp.solve(x_init,u_init,500)
print(ddp.iter)
print(ddp.cost)
ddp.backwardPass()
Vx = np.array(ddp.Vx)
Vxx = np.array(ddp.Vxx)
plt.figure()
pl1, = plt.plot(np.arange(Vx.shape[0]) , Vx[:,20])
# pl2, = plt.plot(np.arange(Vxx.shape[0]) , Vxx[:,20,20])
# plt.legend([pl1,pl2] , ["Vx","Vxx"])
# [state,   heur,   ||x21-dtref||  ,   ||u||12        , previous , friction  , X_bounds ] Augmented
# [state,   heur,   ||x21-dtref||  ,   ||U - dt_ref|| , X_bounds , Cmd_bound , 0        ] Dt step 
# [state,   heur,   ||x21-dtref||  ,    |u|4          , X_bounds , speed_cost , 0        ] Foot Step



# || x21 -dt_ref||^2 not used (x21 = dt)
# Global cost : 
# [state,   heur  ,   ||u||12  , Previous , friction  , X_bounds ,  ||U - dt_ref|| , Cmd_bound  , |u|4 , speed_cost ] #x10
################
# Get results  #
################

Us = ddp.us
Liste = [x for x in Us if (x.size != 4 and x.size != 1) ]
Us =  np.array(Liste)[:,:].transpose()

Xs = np.zeros((21,int(T_gait/dt_mpc)*n_periods + int(np.sum(gait[:2,0])) ))
k = 0
index = 1
for elt in ListAction :
    if elt.__class__.__name__ != "ActionModelQuadrupedStepTime" and  elt.__class__.__name__ != "ActionModelQuadrupedTime": 
        Xs[:,k ] = np.array(ddp.xs[index])
        k = k+1
    index += 1

j = 0
k_cum = 0

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

l_t2 =  [0.]

k_cum = 0 
for i,elt in enumerate(ListDt) :    
    for j in range(int(gait[i,0])) : 
        l_t2.append(l_t2[-1] + elt)

lt_cost = l_t2.copy()
lt_cost.append(lt_cost[-1] + ListDt[-1]) # for temrinal cost
l_t2.pop(0)
l_t2 = np.array(l_t2)

print("Results optim : ")
print("dt1 : " , ListDt[0])
print("dt2 : " , ListDt[2])
print("dt3 : " , ListDt[4])


####################################
## Cost stack
####################################
Names = []
stateCost = np.zeros(len(ListAction) + 1)
forceCost = np.zeros(len(ListAction) + 1)
frictionCost = np.zeros(len(ListAction) + 1)
dt_boundCost = np.zeros(len(ListAction)+1)
deltaFoot = np.zeros(len(ListAction)+1)
speedCost = np.zeros(len(ListAction)+1)
dtCmdBound = np.zeros(len(ListAction)+1)


for index,elt in enumerate(ListAction ):
    if elt.__class__.__name__ == "ActionModelQuadrupedAugmentedTime" :         
        Names.append("A")
        stateCost[index] = elt.Cost[0]
        forceCost[index] = elt.Cost[2]
        frictionCost[index] = elt.Cost[5]
        dt_boundCost[index] = elt.Cost[3]
    elif elt.__class__.__name__ == "ActionModelQuadrupedTime" :
        Names.append("Dt")
        dtCmdBound[index] = elt.Cost[3]
    else : 
        Names.append("Fo")
        speedCost[index] = elt.Cost[3]
        deltaFoot[index] = elt.Cost[2]

Names.append("Terminal")
stateCost[index] = terminalModel.Cost[0]
ind = np.arange(len(ListAction)+1)
plt.figure()
width = 0.6
pl1 = plt.bar(ind, stateCost, width,align='center' , tick_label=Names)
pl2 = plt.bar(ind, forceCost, width, bottom=stateCost,align='center' )
pl3 = plt.bar(ind, frictionCost, width, bottom=forceCost,align='center' )
pl4 = plt.bar(ind, dt_boundCost, width, bottom=frictionCost,align='center' )  
pl5 = plt.bar(ind, deltaFoot, width, bottom=dt_boundCost,align='center' )  
pl6 = plt.bar(ind, speedCost, width, bottom=deltaFoot,align='center' )  
pl7 = plt.bar(ind, dtCmdBound, width, bottom=speedCost,align='center' )
plt.legend([pl1,pl2,pl3,pl4,pl5,pl6,pl7] , ["State cost" , "Norm Force" , "Friction" , "Xdt -+ bounds (aug)"  , "Norm delta foot" , "Speed cost" , "cmd_dt -+ bounds (dt)" ])

plt.figure()
stateCost = np.zeros(len(lt_cost) )
forceCost = np.zeros(len(lt_cost))
frictionCost = np.zeros(len(lt_cost) )
deltaFoot = np.zeros(len(lt_cost))
speedCost = np.zeros(len(lt_cost))
dtCmdBound = np.zeros(len(lt_cost))

gap = 0
x_dt_change = []
x_foot_change = []
y = 0.02
for index,elt in enumerate(ListAction ):
    if elt.__class__.__name__ == "ActionModelQuadrupedAugmentedTime" :  
        stateCost[index-gap] = elt.Cost[0]
        forceCost[index-gap] = elt.Cost[2]
        frictionCost[index-gap] = elt.Cost[5]
    elif elt.__class__.__name__ == "ActionModelQuadrupedTime" :     
        x_dt_change.append(lt_cost[index-gap])
        dtCmdBound[index-gap] = elt.Cost[3]
        gap += 1
        
    else : 
        x_foot_change.append(lt_cost[index-gap])
        speedCost[index-gap] = elt.Cost[3]
        deltaFoot[index-gap] = elt.Cost[2]
        gap += 1
# stateCost[index-gap + 1] = terminalModel.Cost[0]
for elt in x_foot_change : 
    pl7 = plt.axvline(x=elt,color='gray',linestyle='--')

for elt in x_dt_change : 
    pl8 = plt.axvline(x=elt,color='gray',linestyle=(0, (1, 10)) )


pl1, = plt.plot(lt_cost,stateCost , "-x" , color = "b")
pl2, = plt.plot(lt_cost,forceCost , "-x", color = "y")
pl3, = plt.plot(lt_cost,frictionCost , "-x")
pl4, = plt.plot(lt_cost,deltaFoot , "-x" , color = "k")
pl5, = plt.plot(lt_cost,speedCost , "-x" )
pl6, = plt.plot(lt_cost,dtCmdBound , "-x")
plt.legend([pl1,pl2,pl3,pl4,pl5,pl6,pl7,pl8] , ["State cost" , "Norm Force" , "Friction"   , "Norm delta foot" , "Speed cost" , "cmd_dt -+ bounds (dt) " ,
                                                     "Foot step" , "dt change"])
plt.title("Total Cost : " + str(ddp.cost))
txt = "dt min : " + str(mpc_planner_time.dt_min) + "\n" + "dt max : " + str(mpc_planner_time.dt_max) + " \n " + " \n"  
txt += "Optim : \n" + "dt1 : " + str(np.around(ListDt[0],decimals = 4)) + "\ndt2 : " + str(np.around(ListDt[2],decimals = 4))
txt +=  "\ndt3 : " + str(np.around(ListDt[4],decimals = 4)) 
txt += "\n\n\n"
txt += "Vy initial : " + str(V_perturb[1])
txt += "\n"
txt += "dt init : " + str(dt_init)
plt.text(-0.5,0.01,txt)
plt.subplots_adjust(left=0.25)
plt.xlabel("t [s]")
plt.ylabel("cost")

# plt.plot(x_foot_change,y_foot_change,"r--")
# plt.plot()
# print("------------------------")
# print(ddp.Vx[0])
# data = ListAction[0].createData()
# ListAction[0].calc(data, ddp.xs[0] , ddp.us[0] )
# print(data.Lx)
# print(ddp.Vx[1])
# print(ddp.Vxx[0][20,20])
# print(ddp.Vxx[1][20,20])
# print(ddp.Vxx[2][20,20])
# print(ddp.Vxx[3][20,20])
# print(ddp.Vxx[4][20,20])
# print(ddp.Vxx[5][20,20])
# print(ddp.Vxx[6][20,20])


################""


#############
#  Plot     #
#############

# Predicted evolution of state variables
l_t = np.linspace(dt_mpc, T_gait + dt_mpc + dt_mpc*gait[0,0], np.int(T_gait/dt_mpc)*n_periods + int(np.sum(gait[:2,0])))

l_str2 = ["X_ddp", "Y_ddp", "Z_ddp", "Roll_ddp", "Pitch_ddp", "Yaw_ddp", "Vx_ddp", "Vy_ddp", "Vz_ddp", "VRoll_ddp", "VPitch_ddp", "VYaw_ddp"]

index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
plt.figure()
for i in range(12):
    plt.subplot(3, 4, index[i])
       

    if Relaunch_DDP and not Relaunch_TIME: 
        pl1, = plt.plot(l_t, mpc_planner.Xs[i,:], linewidth=2, marker='x')
        plt.legend([pl1] , [l_str2[i]])

    else  : 
        pl1, = plt.plot(l_t, mpc_planner.Xs[i,:], linewidth=2, marker='x')
        pl2, = plt.plot(l_t2, Xs[i,:], linewidth=2, marker='x')
        plt.legend([pl1,pl2] , [l_str2[i] ,  "ddp_time" ])
       

    

# Desired evolution of contact forces
l_str2 = ["FL_X_ddp", "FL_Y_ddp", "FL_Z_ddp", "FR_X_ddp", "FR_Y_ddp", "FR_Z_ddp", "HL_X_ddp", "HL_Y_ddp", "HL_Z_ddp", "HR_X_ddp", "HR_Y_ddp", "HR_Z_ddp"]
index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
plt.figure()
for i in range(12):
    plt.subplot(3, 4, index[i])    
   
    if Relaunch_DDP and not Relaunch_TIME: 
        pl1, = plt.plot(l_t,mpc_planner.Us[i,:], linewidth=2, marker='x')
        plt.legend([pl1] , [l_str2[i] ])

    else : 
        pl1, = plt.plot(l_t,mpc_planner.Us[i,:], linewidth=2, marker='x')
        pl2, = plt.plot(l_t2,Us[i,:], linewidth=2, marker='x')
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
for i in range(4) : 
    if gait[0,i+1] == 1 : 
        plt.plot(fsteps[0,3*i+1] , fsteps[0,3*i+2] , "ko" , markerSize = 7   )
    else : 
        plt.plot(p0[2*i] , p0[2*i+1] , "kx" , markerSize = 8   )

# Position of the center of mass
for elt in Xs : 
    plt.plot(Xs[0,:] ,Xs[1,:] , "gx")


for i in range(4) : 
    for k in range(1,index) : 
        if fsteps[k,3*i+1] != 0. and gait[k,i+1] !=0 and gait[k-1,i+1] == 0  : 
            if i == 0 :
                plt.plot(fsteps[k,3*i+1] , fsteps[k,3*i+2] , "bo" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none'  )
            if i == 1 :
                plt.plot(fsteps[k,3*i+1] , fsteps[k,3*i+2] , "ro" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none' )
            if i == 2 :
                plt.plot(fsteps[k,3*i+1] , fsteps[k,3*i+2] , "ko" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none')
            if i == 3 :
                plt.plot(fsteps[k,3*i+1] , fsteps[k,3*i+2] , "go" , markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none')
    

# for i in range(len(ListAction)) :     
#     x = np.random.rand(20)
#     x2 = np.zeros(21)
#     x2[0:20] = x
#     x2[-1] = 0.02

#     u = np.random.rand(12)
#     if ListAction[i].__class__.__name__ == "ActionModelQuadrupedStepTime" : 
#         u = np.random.rand(4)
#         print("step")

           

#     action =  mpc_planner.ListAction[i]
#     data = action.createData()    
#     action.calc(data,x,u)
#     action.calcDiff(data,x,u)

#     action2 =  ListAction[i]
#     data2 = action2.createData()
#     action2.calc(data2,x2,u)
#     action2.calcDiff(data2,x2,u)

#     # if np.sum(data.xnext - data2.xnext[:20]) != 0 :
#     #     print(i)
#     # print(data.r - data2.r[:24])
#     # print(data.Lx - data2.Lx[:20])
#     # print(data2.xnext)
    
#     if np.sum(np.round(data.cost,decimals = 3) - np.round(data2.cost, decimals = 3)) != 0 :
#         print(data.cost)
#         print(data2.cost)
#         print(i)
#     #     print(i)


# print(np.round(action2.B[9:,:], decimals = 2))
# print(np.round(action.B[9:,:], decimals = 2))



# print(xref[:,:, iteration])
plt.show(block=True)



