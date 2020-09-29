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

folder_name = ""
pathIn = "crocoddyl_eval/test_4/log_eval/trot_064_29x29/"
# res = np.load(pathIn + folder_name + "results_vy_all_true31.npy" , allow_pickle=True )
res = np.load(pathIn + folder_name + "results_vy_all_false29.npy" , allow_pickle=True )
# res = np.load(pathIn + folder_name + "results_wyaw_all_true31.npy" , allow_pickle=True )
# res = np.load(pathIn + folder_name + "results_wyaw_all_false29.npy" , allow_pickle=True )

X = np.linspace(1,-1,29)
Y = np.linspace(-1,1,29)
W = np.linspace(-2.7,2.7,29)

def find_nearest(Vx , Vy):
    idx = (np.abs(X - Vx)).argmin()
    idy = (np.abs(Y - Vy)).argmin()
    # idy = (np.abs(W - Vy)).argmin() # Yaw

    return idx , idy


XX , YY = np.meshgrid(X,W)
Z = np.zeros((XX.shape[0] , YY.shape[1]))
Z_osqp = np.zeros((XX.shape[0] , YY.shape[1]))

plt.figure()

for elt in res : 
    idx , idy = find_nearest(elt[1][0] , elt[1][1])
    # idx , idy = find_nearest(elt[1][0] , elt[1][5]) # Yaw
    Z[idx,idy] = elt[0]



# #### symettry
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
im = plt.imshow(Z ,cmap = plt.cm.binary , extent=(-1,1,-1,1))
plt.xlabel("Lateral Velocity $\dot{p_y} \hspace{2mm} [rad.s^{-1}]$" , fontsize=12)
# plt.xlabel("Yaw rate $\dot{\psi} \hspace{2mm} [rad.s^{-1}]$" , fontsize=12) # Yaw
plt.ylabel("Forward Velocity $\dot{p_x} \hspace{2mm} [m.s^{-1}]$" , fontsize=12)
plt.title("Viable Operating Regions (DDP and foot optimization)" , fontsize=14)





plt.show()