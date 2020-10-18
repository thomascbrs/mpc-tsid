
# coding: utf8
# coding: utf8

import sys
import os 
from sys import argv
sys.path.insert(0, os.getcwd()) # adds current directory to python path

import random
import numpy as np
import matplotlib.pylab as plt; plt.ion()


V0 = 0.0 
acc0 = 0.0
Dt = 0.1
x0 = 0
x1 = 1
 
# Coeff
a0 = x0
a1 = V0 
a2 = acc0/2
a3 = (-3*acc0*Dt**2 - 12*Dt*V0 - 20*x0 + 20*x1)/(2*Dt**3)
a4 = (3*acc0*Dt**2 + 16*Dt*V0 + 30*x0 - 30*x1)/(2*Dt**4)
a5 = (-acc0*Dt**2 - 6*Dt*V0 - 12*x0 + 12*x1)/(2*Dt**5)

def x_5(t) : 
    return a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5

def v_5(t) : 
    return a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4

def a_5(t) : 
    return 2*a2 + 6*a3*t + 12*a4*t**2 + 20*a5*t**3




t = np.linspace(0.0,Dt,100)
plt.figure(1)
plt.title("V(t)")
plt.plot(t,v_5(t))


V0 = 10.0 
acc0 = 100
Dt = 0.1
x0 = 0
x1 = 1
 
# Coeff
a0 = x0
a1 = V0 
a2 = acc0/2
a3 = (-3*acc0*Dt**2 - 12*Dt*V0 - 20*x0 + 20*x1)/(2*Dt**3)
a4 = (3*acc0*Dt**2 + 16*Dt*V0 + 30*x0 - 30*x1)/(2*Dt**4)
a5 = (-acc0*Dt**2 - 6*Dt*V0 - 12*x0 + 12*x1)/(2*Dt**5)
plt.plot(t,v_5(t))