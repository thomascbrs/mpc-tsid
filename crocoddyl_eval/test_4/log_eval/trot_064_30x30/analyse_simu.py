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
pathIn = "crocoddyl_eval/test_4/log_eval/trot_064_30x30/"
# res = np.load(pathIn + folder_name + "results_vy_all_true2.npy" , allow_pickle=True )
res = np.load(pathIn + folder_name + "results_wyaw_all_true27.npy" , allow_pickle=True )

X = np.linspace(1,-1,30)
Y = np.linspace(-1,1,30)
W = np.linspace(-2.7,2.7,30)

def find_nearest(Vx , Vy):
    idx = (np.abs(X - Vx)).argmin()
    idy = (np.abs(W - Vy)).argmin()

    return idx , idy


XX , YY = np.meshgrid(X,W)
Z = np.zeros((XX.shape[0] , YY.shape[1]))
Z_osqp = np.zeros((XX.shape[0] , YY.shape[1]))

plt.figure()

for elt in res : 
    idx , idy = find_nearest(elt[1][0] , elt[1][5])
    Z[idx,idy] = elt[0]



#### symettry
for i in range(30) : 
    for j in range(15) : 
        if Z[i,j] != Z[i,-j-1] : 
            Z[i,j] = False
            Z[i,-j-1] = False

for i in range(30) : 
    for j in range(15) : 
        if Z[j,i] != Z[-j-1,i] : 
            print("--")
            print(i)
            print(j)
            print(-j-1)
            Z[j,i] = False
            Z[-j-1,i] = False
    

plt.rc('text', usetex=True)
im = plt.imshow(Z ,cmap = plt.cm.binary , extent=(-2.7,2.7,-1,1))
plt.xlabel("Lateral Velocity $\dot{p_y} \hspace{2mm} [m.s^{-1}]$" , fontsize=12)
plt.ylabel("Forward Velocity $\dot{p_x} \hspace{2mm} [m.s^{-1}]$" , fontsize=12)
plt.title("Viable Operating Regions (DDP and foot optimization)" , fontsize=14)





plt.show()