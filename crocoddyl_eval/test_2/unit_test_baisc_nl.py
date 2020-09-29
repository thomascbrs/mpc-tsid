# coding: utf8
# coding: utf8

import sys
import os 
from sys import argv
sys.path.insert(0, os.getcwd()) # adds current directory to python path

import numpy as np

import matplotlib.pylab as plt
import quadruped_walkgen 
import crocoddyl



N_trial = 1
a = 1 
b = -1

epsilon = 1e-3

actionModel = quadruped_walkgen.ActionModelQuadrupedNonLinear()
actionModel.stateWeights = np.zeros(12)
data = actionModel.createData()
actionModel.shoulderWeights = 1
actionModel.shoulder_hlim = 0.245
    # x = a + (b-a)*np.random.rand(12)
x = np.array([0.,0.,0.24, 9/57,5/57,0., 0.,0.,0. ,0.,0.,0.])

u = a + (b-a)*np.random.rand(12)

l_feet = np.random.rand(3,4)
xref = np.array([0.,0.,0.24, 9/57,5/57,0., 0.,0.,0. ,0.,0.,0.])
S = np.array([1,1,1,1])
actionModel.updateModel(l_feet , xref , S )
actionModel.calc(data , x , u )
actionModel.calcDiff(data , x , u )

print(data.Lxx[:6,:6])
print(data.Lx)