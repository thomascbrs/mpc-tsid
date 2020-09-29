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

folder_name = "064_ddp_heur_linear/"
pathIn = "crocoddyl_eval/test_4/log_eval/"
# Pb with the xref 
# res = np.load(pathIn + folder_name + "results_wyaw_ddp_heur_linearModel.npy" , allow_pickle=True )
# res = np.load(pathIn + folder_name + "results_vy_ddp_heur_linearModel.npy" , allow_pickle=True )
# Pb fixed 
# res = np.load(pathIn + folder_name + "results_wyaw_ddp_heur_LinearModel2.npy" , allow_pickle=True )
res = np.load(pathIn + folder_name + "results_vy_ddp_heur_LinearModel2.npy" , allow_pickle=True )

Vy_analysis = False # False --> Yaw
symmetry = True 

if Vy_analysis :
    res = np.load(pathIn + folder_name + "results_vy_ddp_heur_LinearModel3.npy" , allow_pickle=True )
else : 
    res = np.load(pathIn + folder_name + "results_wyaw_ddp_heur_LinearModel3.npy" , allow_pickle=True )

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
Z_osqp = np.zeros((XX.shape[0] , YY.shape[1]))

plt.figure()

for elt in res : 
    if Vy_analysis : 
        idx , idy = find_nearest(elt[1][0] , elt[1][1])
        if elt[1][1] < 0.6 and elt[1][1] > 0.4 and elt[1][0] == 0.0 : 
            print(elt[1][1])
            print(elt[0])
    else : 
        idx , idy = find_nearest(elt[1][0] , elt[1][5]) # Yaw
    Z[idx,idy] = elt[0]



#### symmetry
if symmetry : 
    for i in range(29) : 
        for j in range(14) : 
            if Z[i,j] != Z[i,-j-1] : 
                Z[i,j] = False
                Z[i,-j-1] = False

    for i in range(29) : 
        for j in range(14) : 
            if Z[j,i] != Z[-j-1,i] : 
                print("--")
                print(i)
                print(j)
                print(-j-1)
                Z[j,i] = False
                Z[-j-1,i] = False
    

plt.rc('text', usetex=True)
if Vy_analysis : 
    im = plt.imshow(Z ,cmap = plt.cm.binary , extent=(-1,1,-1,1))
    plt.xlabel("Lateral Velocity $\dot{p_y} \hspace{2mm} [rad.s^{-1}]$" , fontsize=12)
else :
    im = plt.imshow(Z ,cmap = plt.cm.binary , extent=(-2.7,2.7,-1,1)) 
    plt.xlabel("Yaw rate $\dot{\psi} \hspace{2mm} [rad.s^{-1}]$" , fontsize=12) 

plt.ylabel("Forward Velocity $\dot{p_x} \hspace{2mm} [m.s^{-1}]$" , fontsize=12)
plt.title("Viable Operating Regions (DDP + Heuristic)" , fontsize=14)





plt.show()