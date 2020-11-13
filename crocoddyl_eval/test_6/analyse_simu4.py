
# coding: utf8
# coding: utf8

import sys
import os 
from sys import argv
sys.path.insert(0, os.getcwd()) # adds current directory to python path

import random
import numpy as np
import matplotlib.pylab as plt; plt.ion()
import numpy as np
import numpy.polynomial.polynomial as nppol
from sympy import init_printing, Symbol, expand 
from sympy import poly
from sympy.abc import x
import sympy

# V0 = Symbol('V0' , positive = True)
# acc0 = Symbol('acc0', positive = True)
# Dt = Symbol('Dt', positive = True)
# h = Symbol('h' , positive = True)
# x0 = 0
# x1 = 1
 
# # Coeff
# a0 = x0
# a1 = V0 
# a2 = acc0/2
# a3 = (-3*acc0*Dt**2 - 12*Dt*V0 - 20*x0 + 20*x1)/(2*Dt**3)
# a4 = (3*acc0*Dt**2 + 16*Dt*V0 + 30*x0 - 30*x1)/(2*Dt**4)
# a5 = (-acc0*Dt**2 - 6*Dt*V0 - 12*x0 + 12*x1)/(2*Dt**5)

# V =  poly(a1 + 2*a2*x + 3*a3*x**2 + 4*a4*x**3 + 5*a5*x**4 )



def x_5(t) : 
    return a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5

def v_5(t) : 
    return a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4

def a_5(t) : 
    return 2*a2 + 6*a3*t + 12*a4*t**2 + 20*a5*t**3




# t = np.linspace(0.0,Dt,100)
# plt.figure(1)
# plt.title("V(t)")
# plt.plot(t,v_5(t))

def v_h(h) : 
    x = h
    b1 = Dt*acc0*(x-(9/2)*x**2 + 6*x**3 - (5/2)*x**4) 
    b2 = V0*(1-18*x**2 + 32*x**3 - 15*x**4)
    b3 = (x1-x0)/Dt * (30*x**2 - 60*x**3 + 30*x**4)
    return b1 + b2 + b3



V0 = -10.0 
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


t = np.linspace(0.0,Dt,100)
plt.plot(t,v_5(t))

h = np.linspace(0.01,1,10)
vh = v_h(h)
plt.plot(Dt*h,vh,"x")