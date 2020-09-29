# coding: utf8
# coding: utf8

import sys
import os 
from sys import argv
sys.path.insert(0, os.getcwd()) # adds current directory to python path


import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
####################
# Recovery of Data
####################

Vy_analysis = True # False --> Yaw
symmetry = False 
opacity = 0.6

folder_name = "064_ddp_heur_nonLin/"
folder_name_2 = "064_ddp_heur_linear/"
folder_name_3 = "064_ddp_fsteps/"

pathIn = "crocoddyl_eval/test_4/log_eval/"

# res  -> Non linear, dark area
# res2 -> linear, red area

if Vy_analysis : 
    
    # lin VS NL VS Fsteps
    # res = np.load(pathIn + folder_name + "vy/" + "results_vy_ddp_heur_nonLinearModel3.npy" , allow_pickle=True ) 
    # res2 = np.load(pathIn + folder_name_2 + "vy/" + "results_vy_ddp_heur_LinearModel3.npy" , allow_pickle=True )
    # res3 = np.load(pathIn + folder_name_3 + "vy/" + "results_vy_ddp_fstep_nosh.npy" , allow_pickle=True )

    # lin vs NL vs fsteps: shoulder cost
    res = np.load(pathIn + folder_name + "vy/" + "results_vy_ddp_nl_sh_0225_2.npy" , allow_pickle=True ) 
    res2 = np.load(pathIn + folder_name_2 + "vy/" + "results_vy_ddp_lin_sh_0225_2.npy" , allow_pickle=True )
    res3 = np.load(pathIn + folder_name_3 + "vy/" + "results_vy_ddp_fstep_sh_0225_2.npy" , allow_pickle=True )

else : 
    # lin VS NL VS Fsteps
    res = np.load(pathIn + folder_name + "wyaw/" +  "results_wyaw_ddp_heur_nonLinearModel3.npy" , allow_pickle=True )
    res2 = np.load(pathIn + folder_name_2 + "wyaw/" + "results_wyaw_ddp_heur_LinearModel3.npy" , allow_pickle=True )
    res3 = np.load(pathIn + folder_name_3 + "wyaw/" + "results_wyaw_ddp_fstep_nosh.npy" , allow_pickle=True )

    # lin VS NL vs ftspes: shoulder cost
    # res = np.load(pathIn + folder_name + "wyaw/" +  "results_wyaw_ddp_nl_sh_0225_2.npy" , allow_pickle=True )
    # res2 = np.load(pathIn + folder_name_2 + "wyaw/" + "results_wyaw_ddp_lin_sh_0225_2.npy" , allow_pickle=True )
    # res3 = np.load(pathIn + folder_name_3 + "wyaw/" + "results_wyaw_ddp_fstep_sh_0225_2.npy" , allow_pickle=True )

X = np.linspace(1,-1,29)
Y = np.linspace(-1,1,29)
W = np.linspace(-2.7,2.7,29)

def find_nearest(Vx , Vy):
    idx = (np.abs(X - Vx)).argmin()
    if Vy_analysis : 
        idy = (np.abs(Y - Vy)).argmin()
    else  :
        idy = (np.abs(W - Vy)).argmin() 

    return idx , idy


XX , YY = np.meshgrid(X,W)
Z = np.zeros((XX.shape[0] , YY.shape[1]))
Z2 = np.zeros((XX.shape[0] , YY.shape[1]))
Z3 = np.zeros((XX.shape[0] , YY.shape[1]))
alphas2 = np.zeros((XX.shape[0] , YY.shape[1])) # lin
alphas = np.zeros((XX.shape[0] , YY.shape[1])) # nl

# Non linear
# Z --> Non linear
for elt in res : 
    if Vy_analysis : 
        idx , idy = find_nearest(elt[1][0] , elt[1][1])
    else : 
        idx , idy = find_nearest(elt[1][0] , elt[1][5]) # Yaw

    # if not Vy_analysis and elt[1][0] >= 0.78 and  elt[1][5] >= 0.19 : # check video, fall at the end
    #     Z[idx,idy] = False
    # elif not Vy_analysis and elt[1][0] >= 0.71 and  elt[1][5] >= 0.38 : # check video, fall at the end
    #     Z[idx,idy] = False
    # else : 
    #     Z[idx,idy] = elt[0]

    
   
    Z[idx,idy] = elt[0]

# Linear
# Z2 --> linear
for elt in res2 : 
    if Vy_analysis : 
        
        idx , idy = find_nearest(elt[1][0] , elt[1][1])
    else : 
        idx , idy = find_nearest(elt[1][0] , elt[1][5]) # Yaw

    # Linear shoulder Vy, shoulder cost, correction
    # if Vy_analysis and elt[1][1] >= 0.41 : # check video, fall at the end
    #     Z2[idx,idy] = False
    # elif not Vy_analysis and elt[1][5] >= 0.38 and elt[1][0] >= 0.6 : 
    #     Z2[idx,idy] = False
    # elif Vy_analysis and elt[1][1] >= 0.35 and elt[1][0] >= 0.21: 
    #     Z2[idx,idy] = False
    # else : 
    #     Z2[idx,idy] = elt[0]

    # Linear shoulder wyaw, shoulder cost, correction
    # if not Vy_analysis and elt[1][5] >= 0.38 and elt[1][0] >= 0.6 : # check video, fall at the end
    #     Z2[idx,idy] = False
    # else : 
    #     Z2[idx,idy] = elt[0]
    
    Z2[idx,idy] = elt[0]

# Z3 --> fsteps
for elt in res3 : 
    if Vy_analysis : 
        idx , idy = find_nearest(elt[1][0] , elt[1][1])
    else : 
        idx , idy = find_nearest(elt[1][0] , elt[1][5]) # Yaw

    # fstep shoulder Vy, shoulder cost, correction
    # if not Vy_analysis and elt[1][0] >= 0.06 and elt[1][5] >= 2.47   : # check video, fall at the end
    #     Z3[idx,idy] = False
    # elif not Vy_analysis and elt[1][0] >= 0.59 and elt[1][5] >= 0.45 : 
    #     Z3[idx,idy] = False
    # elif Vy_analysis and elt[1][0] >= 0.9 : 
    #     Z3[idx,idy] = False
    # elif Vy_analysis and elt[1][0] >= 0.8 and elt[1][1] >= 0.22: 
    #     Z3[idx,idy] = False
    # else : 
    #     Z3[idx,idy] = elt[0]
    
   
    Z3[idx,idy] = elt[0]



##############
## Symmetry
##############
if symmetry : 
    for i in range(29) : 
        for j in range(14) : 
            if Z[i,j] != Z[i,-j-1] : 
                Z[i,j] = False
                Z[i,-j-1] = False
            if Z2[i,j] != Z2[i,-j-1] : 
                Z2[i,j] = False
                Z2[i,-j-1] = False
            
            if Z3[i,j] != Z3[i,-j-1] : 
                Z3[i,j] = False
                Z3[i,-j-1] = False

    for i in range(29) : 
        for j in range(14) : 
            if Z[j,i] != Z[-j-1,i] : 
                Z[j,i] = False
                Z[-j-1,i] = False
            if Z2[j,i] != Z2[-j-1,i] : 
                Z2[j,i] = False
                Z2[-j-1,i] = False
            if Z3[j,i] != Z3[-j-1,i] : 
                Z3[j,i] = False
                Z3[-j-1,i] = False

for i in range(Z.shape[0]) : 
    for j in range(Z.shape[1]) : 
        if Z2[i,j] == True:
            alphas2[i,j] = 0.7

for i in range(Z.shape[0]) : 
    for j in range(Z.shape[1]) : 
        if Z[i,j] == True:
            alphas[i,j] = 0.6
            

colors = [(1, 1, 1), (0.9, 0.0, 0.0)]  # R -> G -> B
# n_bins = [3, 6, 10, 100]  # Discretizes the interpolation into bins
cmap_name = 'my_list'
# Create the colormap
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=10)


colors = [(1, 1, 1), (1., 0.8, 0.1)]  # R -> G -> B
# n_bins = [3, 6, 10, 100]  # Discretizes the interpolation into bins
cmap_name = 'my_list_2'
# Create the colormap
cm2 = LinearSegmentedColormap.from_list(cmap_name, colors, N=10)



# Graph Non linear
# plt.figure()
# plt.rc('text', usetex=True)
# if Vy_analysis : 
#     im = plt.imshow(Z ,cmap = plt.cm.binary , extent=(-1,1,-1,1))
#     plt.xlabel("Lateral Velocity $\dot{p_y} \hspace{2mm} [rad.s^{-1}]$" , fontsize=12)
# else :
#     im = plt.imshow(Z ,cmap = plt.cm.binary , extent=(-2.7,2.7,-1,1)) 
#     plt.xlabel("Yaw rate $\dot{\psi} \hspace{2mm} [rad.s^{-1}]$" , fontsize=12) 

# plt.ylabel("Forward Velocity $\dot{p_x} \hspace{2mm} [m.s^{-1}]$" , fontsize=12)
# plt.title("Viable Operating Regions (DDP Non Linear + Heuristic)" , fontsize=14)

# # Graph Linear
# plt.figure()
# plt.rc('text', usetex=True)
# if Vy_analysis : 
#     im = plt.imshow(Z2 ,cmap = plt.cm.binary , extent=(-1,1,-1,1))
#     plt.xlabel("Lateral Velocity $\dot{p_y} \hspace{2mm} [rad.s^{-1}]$" , fontsize=12)
# else :
#     im = plt.imshow(Z2 ,cmap = plt.cm.binary , extent=(-2.7,2.7,-1,1)) 
#     plt.xlabel("Yaw rate $\dot{\psi} \hspace{2mm} [rad.s^{-1}]$" , fontsize=12) 

# plt.ylabel("Forward Velocity $\dot{p_x} \hspace{2mm} [m.s^{-1}]$" , fontsize=12)
# plt.title("Viable Operating Regions (DDP + Heuristic)" , fontsize=14)


patches =[mpatches.Patch(color=[0,0,0],label="Fstep") , mpatches.Patch(color=[0.9, 0.0, 0.0,0.6],label="NL") , mpatches.Patch(color=[1, 0.8, 0.0,0.7],label="Linear")   ]


# Mix
plt.figure()
plt.rc('text', usetex=True)
if Vy_analysis : 
    im = plt.imshow(Z3 ,cmap = plt.cm.binary , extent=(-1,1,-1,1) )

    im = plt.imshow(Z2 ,cmap = cm2 , extent=(-1,1,-1,1) , alpha = alphas2  )
    im = plt.imshow(Z ,cmap = cm , extent=(-1,1,-1,1) , alpha = alphas )
    plt.xlabel("Lateral Velocity $\dot{p_y} \hspace{2mm} [rad.s^{-1}]$" , fontsize=12)
else :
    # im = plt.imshow(Z ,cmap = plt.cm.binary , extent=(-2.7,2.7,-1,1) )
    # im = plt.imshow(Z2 ,cmap = cm2 ,  extent=(-2.7,2.7,-1,1) , alpha = alphas2  )
    im = plt.imshow(Z3 ,cmap = plt.cm.binary , extent=(-2.7,2.7,-1,1) )

    im = plt.imshow(Z2 ,cmap = cm2 , extent=(-2.7,2.7,-1,1)  , alpha = alphas2  )
    im = plt.imshow(Z ,cmap = cm , extent=(-2.7,2.7,-1,1) , alpha = alphas )
    plt.xlabel("Yaw rate $\dot{\psi} \hspace{2mm} [rad.s^{-1}]$" , fontsize=12) 

plt.ylabel("Forward Velocity $\dot{p_x} \hspace{2mm} [m.s^{-1}]$" , fontsize=12)
plt.title("Viable Operating Regions (Linear, NL and Fsteps)" , fontsize=14)
# plt.grid()

plt.legend(handles=patches)

plt.show()