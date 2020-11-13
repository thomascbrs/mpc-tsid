# coding: utf8
# coding: utf8

import sys
import os 
from sys import argv
sys.path.insert(0, os.getcwd()) # adds current directory to python path


import numpy as np
import matplotlib.pylab as plt ; plt.ion()
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
####################
# Recovery of Data
####################

Vy_analysis = True # False --> Yaw
symmetry = False 
opacity = 0.6

pathIn = "crocoddyl_eval/test_8/log_eval/walk_linear_mpc/"

qtsid_full = np.load(pathIn + "qtsid.npy" , allow_pickle=True ) 
vtsid_full = np.load(pathIn + "vtsid.npy" , allow_pickle=True ) 
torques_ff = np.load(pathIn + "torques_ff.npy" , allow_pickle=True ) 

qtsid = qtsid_full[7:,:]
vtsid = vtsid_full[6:,:]

plt.figure() 
plt.suptitle("qtsid")
for i in range(12) : 
    plt.subplot(4,3,i+1)
    plt.plot(qtsid[i,:] )

plt.figure() 
plt.suptitle("vtsid")
for i in range(12) : 
    plt.subplot(4,3,i+1)
    plt.plot(vtsid[i,:] )

plt.figure() 
plt.suptitle("torques_ff")
for i in range(12) : 
    plt.subplot(4,3,i+1)
    plt.plot(torques_ff[i,:] )