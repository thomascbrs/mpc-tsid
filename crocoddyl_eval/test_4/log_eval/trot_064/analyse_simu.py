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

folder_name = "trot_064/"
pathIn = "crocoddyl_eval/test_4/log_eval/"
res = np.load(pathIn + folder_name + "results_vy.npy" , allow_pickle=True )
res_osqp = np.load(pathIn + folder_name + "results_osqp.npy" , allow_pickle=True )

import numpy as np

X = np.linspace(1,-1,25)
Y = np.linspace(-1,1,25)


def find_nearest(Vx , Vy):
    idx = (np.abs(X - Vx)).argmin()
    idy = (np.abs(Y - Vy)).argmin()

    return idx , idy


XX , YY = np.meshgrid(X,Y)
Z = np.zeros((XX.shape[0] , YY.shape[1]))
Z_osqp = np.zeros((XX.shape[0] , YY.shape[1]))
# plt.figure()

# for elt in res : 
#     if elt[0] == True : 
#         plt.plot(elt[1][0] , elt[1][1] , "bs" , markerSize= "13")
#     else :
#         pass

# plt.xlim([-1,1])
# plt.ylim([-1,1])

plt.figure()

for elt in res : 
    idx , idy = find_nearest(elt[1][0] , elt[1][1])
    if (idx == 9 and idy == 7) or  (idx == 15 and idy == 17): 
        Z[idx,idy] = True # new gain on shoulder weight on y axis, does not chnage other results
    else : 
        Z[idx,idy] = elt[0]

for elt in res_osqp :   
    idx , idy = find_nearest(elt[1][0] , elt[1][1])
    if (idx == 16 and idy == 16) or (idx == 8  and idy == 16)  or (idx == 15 and idy == 17) or (idx == 9  and idy == 17) or (idx == 12  and idy == 6)  :
        Z_osqp[idx,idy] = False # check on simu ; limb switch direction, simu duration not long enough
    else : 
        Z_osqp[idx,idy] = elt[0]

print(find_nearest(-0.25 , 0.416667))
print(find_nearest(0.25 , -0.416667))

plt.rc('text', usetex=True)
im = plt.imshow(Z ,cmap = plt.cm.binary , extent=(-1,1,-1,1))
plt.xlabel("Lateral Velocity $\dot{p_y} \hspace{2mm} [m.s^{-1}]$" , fontsize=12)
plt.ylabel("Forward Velocity $\dot{p_x} \hspace{2mm} [m.s^{-1}]$" , fontsize=12)
plt.title("Viable Operating Regions" , fontsize=14)


plt.figure()
plt.rc('text', usetex=True)
im = plt.imshow(Z_osqp ,cmap = plt.cm.binary , extent=(-1,1,-1,1))
plt.xlabel("Lateral Velocity $\dot{p_y} \hspace{2mm} [m.s^{-1}]$" , fontsize=12)
plt.ylabel("Forward Velocity $\dot{p_x} \hspace{2mm} [m.s^{-1}]$" , fontsize=12)
plt.title("Viable Operating Regions" , fontsize=14)

plt.show()