# coding: utf8
# coding: utf8

import sys
import os 
from sys import argv
sys.path.insert(0, os.getcwd()) # adds current directory to python path


import numpy as np
import matplotlib.pylab as plt

####################
# Recovery of Data
####################

folder_name = "test_2/"
pathIn = "crocoddyl_eval/test_5/log_eval/"
oC = np.load(pathIn + folder_name + "oC.npy" , allow_pickle=True )
o_feet_ = np.load(pathIn + folder_name + "o_feet_.npy" , allow_pickle=True )
o_feet_heur = np.load(pathIn + folder_name + "o_feet_heur.npy" , allow_pickle=True )
gait_ = np.load(pathIn + folder_name + "gait_.npy" , allow_pickle=True )
pred_trajectories = np.load(pathIn + folder_name + "pred_trajectories.npy" , allow_pickle=True )
l_feet_ = np.load(pathIn + folder_name + "l_feet_.npy" , allow_pickle=True )
feet_pos = np.load(pathIn + folder_name + "feet_pos.npy" , allow_pickle=True )
feet_pos_target = np.load(pathIn + folder_name + "feet_pos_target.npy" , allow_pickle=True )
pos_pred_local = np.load(pathIn + folder_name + "pos_pred_local.npy" , allow_pickle=True )
pos_pred_local_heur = np.load(pathIn + folder_name + "pos_pred_local_heur.npy" , allow_pickle=True )

plt.figure()
iteration_begin = 300
iteration = 315
# iteration_begin = 207
# iteration = 222
# iteration_begin = 400
# iteration = 463
# iteration_begin = 480
# iteration = 479
print(gait_[:,:,iteration_begin])

p0 = [ 0.1946,0.14695, 0.1946,-0.14695, -0.1946,   0.14695 ,-0.1946,  -0.14695]
plt.grid()

d = 15
for i in range(iteration_begin ,  iteration + 1) : 
    pl1, = plt.plot(oC[0,20*i]  , oC[1,20*i] , 'gx' , markerSize = 8)

for i in range(4) : 
    pl2, = plt.plot(o_feet_[0,i,iteration_begin - 1 ] , o_feet_[1,i ,iteration_begin - 1 ]  , "ro", markerSize = 12)

k = 1
for i in range(iteration_begin , iteration) : 
    for j in range(4) : 
        if gait_[0,j+1,i] == 0 : 
            if k == 2 : 
                pl3, = plt.plot(o_feet_[0,j,i ] , o_feet_[1,j,i ] , "ko", markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none' )
                pl4, = plt.plot(o_feet_heur[0,j,i ] , o_feet_heur[1,j,i ] , "bo", markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none' )
            elif k == 15 : 
                pl5, = plt.plot(o_feet_[0,j,i ] , o_feet_[1,j,i ] , "ko", markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none' )
                pl6, = plt.plot(o_feet_heur[0,j,i ] , o_feet_heur[1,j,i ] , "bo", markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none' )

            else : 
                plt.plot(o_feet_[0,j,i ] , o_feet_[1,j,i ] , "ko", markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none' )
                plt.plot(o_feet_heur[0,j,i ] , o_feet_heur[1,j,i ] , "bo", markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none' )

            pl7, = plt.plot(feet_pos[0,j,i*20] , feet_pos[1,j,i*20] ,  "rx",  markerSize = 7  )
        # pl6, = plt.plot(feet_pos_target[0,j,i*20] , feet_pos_target[1,j,i*20] ,  "ko", markerSize = "5"   )
    
    k += 1

plt.arrow(0.7, 0, 0.05, 0.0 , lw = 0.02 , fc = "k" , head_width = 0.008)
plt.text(0.7, 0.025 , "Walking Direction" , fontsize = 10)


# plt.rc('text', usetex=True)
plt.title("Flying phase, DDP + Feet optimisation, 15 control cycles, world frame"  , fontsize=14)
plt.legend([pl1 , pl2 ,pl3 ,pl4 ,pl5,pl6,pl7 ] , ["CoM" , "Initial feet positions" , "ddp decision, cycle 1" , "heuristic decision, cycle 1" ,  "ddp decision, cycle 15" , "heuristic decision, cycle 15" ,"ground projection of flying feet"] , loc = 6)


# for i in range(4) : 
#     plt.plot(o_feet_[0,i,iteration  ] , o_feet_[1,i ,iteration  ]  , "ro", markerSize = 8)

######################
# Local frame
######################
# plt.figure()
# for i in range(iteration_begin , iteration) : 
#     plt.plot(pred_trajectories[0,0,i]  , pred_trajectories[1,0,i] , 'kx' , markerSize = 8)

# for i in range(4) : 
#     plt.plot(l_feet_[0,i,iteration_begin - 1 ] , l_feet_[1,i ,iteration_begin - 1 ]  , "ro", markerSize = 8)
    


# k =1
# for i in range(iteration_begin , iteration) : 
#     for j in range(4) : 
#         if gait_[0,j+1,i] == 0 : 
#             plt.plot(pos_pred_local[0,j,i ] , pos_pred_local[1,j,i ] , "go", markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none' )
#             plt.plot(pos_pred_local_heur[0,j,i ] , pos_pred_local_heur[1,j,i ] , "bo", markerSize = int(20/np.sqrt(k)) ,  markerfacecolor='none' )

#             # plt.plot(p0[2*j] + pred_trajectories[0,iteration - i , i], p0[2*j+1] +  pred_trajectories[1,iteration - i , i] , "ks" , markerSize = 10 , markerfacecolor='none' )
        
       
#         # plt.plot(feet_pos[0,j,i*20] , feet_pos[1,j,i*20] ,  "yo", markerSize = "10"   )
#         # plt.plot(feet_pos_target[0,j,i*20] , feet_pos_target[1,j,i*20] ,  "ko", markerSize = "5"   )
    
#     k += 1


# plt.plot()

# for i in range(4) : 
#     plt.plot(o_feet_[0,i,iteration  ] , o_feet_[1,i ,iteration  ]  , "ro", markerSize = 8)


plt.show()