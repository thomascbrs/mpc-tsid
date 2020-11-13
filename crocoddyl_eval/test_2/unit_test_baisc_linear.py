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



N_trial = 500
a = 1 
b = -1
c = -5
d = 50

epsilon = 1e-3


################################################
## CHECK DERIVATIVE WITH NUM_DIFF , 
#################################################

actionModel = quadruped_walkgen.ActionModelQuadruped()
actionModel.shoulderWeights = 10
actionModel.shoulder_hlim = 0.25
model_diff = crocoddyl.ActionModelNumDiff(actionModel)
data = actionModel.createData()
dataCpp = model_diff.createData()

# RUN CALC DIFF
def run_calcDiff_numDiff(epsilon) :
  Lx = 0
  Lx_err = 0
  Lu = 0
  Lu_err = 0
  Lxx = 0
  Lxx_err = 0
  Luu = 0
  Luu_err = 0
  Luu_noFri = 0
  Luu_err_noFri = 0
  Fx = 0
  Fx_err = 0 
  Fu = 0
  Fu_err = 0    

  for k in range(N_trial):    

    x = a + (b-a)*np.random.rand(12)
    x[2] = 0.24
    # x = np.array([0.,0.,0.24, 8/57,1/57,0., 0.,0.,0. ,0.,0.,0.])
    u = c + (d-c)*np.random.rand(12)

    # u = np.array([1,1,2,1,1,2,1,1,2,1,1,-30])
    # print(u)

    l_feet = np.random.rand(3,4)
    xref = np.random.rand(12)
    nb = 0.8
    if nb < 0.333 : 
      S = np.array([1,1,1,1])
    elif nb >= 0.333 and nb < 0.666 :
      S = np.array([0,1,1,0])
    else : 
      S = np.array([1,0,0,1])

    actionModel.updateModel(l_feet , xref , S )
    # actionModel.lastPositionWeights = np.full(8,0.0)
    model_diff = crocoddyl.ActionModelNumDiff(actionModel)
    dataCpp = model_diff.createData()
    
   
    # Run calc & calcDiff function : numDiff 
    
    model_diff.calc(dataCpp , x , u )
    model_diff.calcDiff(dataCpp , x , u )
    
    # Run calc & calcDiff function : c++
    actionModel.calc(data , x , u )
    actionModel.calcDiff(data , x , u )

    Lx +=  np.sum( abs((data.Lx - dataCpp.Lx )) >= epsilon  ) 
    Lx_err += np.sum( abs((data.Lx - dataCpp.Lx )) )  
    print(data.Lu)
    print(dataCpp.Lu)
    print(data.Luu)
    print(dataCpp.Luu)



    Lu +=  np.sum( abs((data.Lu - dataCpp.Lu )) >= epsilon  ) 
    Lu_err += np.sum( abs((data.Lu - dataCpp.Lu )) )  

    Lxx +=  np.sum( abs((data.Lxx - dataCpp.Lxx )) >= epsilon  ) 
    Lxx_err += np.sum( abs((data.Lxx - dataCpp.Lxx )) )  

    Luu +=  np.sum( abs((data.Luu - dataCpp.Luu )) >= epsilon  ) 
    Luu_err += np.sum( abs((data.Luu - dataCpp.Luu )) ) 

    Fx +=  np.sum( abs((data.Fx - dataCpp.Fx )) >= epsilon  ) 
    Fx_err += np.sum( abs((data.Fx - dataCpp.Fx )) )  

    Fu +=  np.sum( abs((data.Fu - dataCpp.Fu )) >= epsilon  ) 
    Fu_err += np.sum( abs((data.Fu - dataCpp.Fu )) )  

    # No friction cone : 
    # actionModel.frictionWeights = 0.0    
    # model_diff = crocoddyl.ActionModelNumDiff(actionModel)   
   
    # # Run calc & calcDiff function : numDiff     
    # model_diff.calc(dataCpp , x , u )
    # model_diff.calcDiff(dataCpp , x , u )
    
    # # Run calc & calcDiff function : c++
    # actionModel.calc(data , x , u )
    # actionModel.calcDiff(data , x , u )
    Luu_noFri +=  np.sum( abs((data.Luu - dataCpp.Luu )) >= epsilon  ) 
    Luu_err_noFri += np.sum( abs((data.Luu - dataCpp.Luu )) ) 


  
  Lx_err = Lx_err /N_trial
  Lu_err = Lu_err/N_trial
  Lxx_err = Lxx_err/N_trial    
  Luu_err = Luu_err/N_trial
  Fx_err = Fx_err/N_trial
  Fu_err = Fu_err/N_trial
  
  return Lx , Lx_err , Lu , Lu_err , Lxx , Lxx_err , Luu , Luu_err, Luu_noFri , Luu_err_noFri, Fx, Fx_err, Fu , Fu_err


Lx , Lx_err , Lu , Lu_err , Lxx , Lxx_err , Luu , Luu_err , Luu_noFri , Luu_err_noFri , Fx, Fx_err, Fu , Fu_err = run_calcDiff_numDiff(epsilon)

print("\n \n")
print(" Action Model Quadruped, linear inertial model (x = 12 , u = 12) ")
print(" \n ------------------------------------------ " )
print(" Checking implementation of the derivatives ")
print(" Using crocoddyl NumDiff class")
print(" ------------------------------------------ " )

print("\n Luu is calculated with the residual cost and can not be exact")
print(" Because of the friction cost term")
print("\n")
print("Espilon : %f" %epsilon)
print("N_trial : %f" %N_trial)
print("\n")

if Fx == 0:  print("Fx : OK    (error : %f)" %Fx_err)
else :     print("Fx : NOT OK !!!   (error : %f)" %Fx_err)

if Fu == 0:  print("Fu : OK    (error : %f)" %Fu_err)
else :     print("Fu : NOT OK !!!   (error : %f)" %Fu_err)
if Lx == 0:  print("Lx : OK    (error : %f)" %Lx_err)
else :     print("Lx : NOT OK !!!    (error : %f)" %Lx_err )

if Lu == 0:  print("Lu : OK    (error : %f)" %Lu_err)
else :     print("Lu : NOT OK !!!    (error : %f)" %Lu_err)

if Lxx == 0:  print("Lxx : OK    (error : %f)" %Lxx_err)
else :     print("Lxx : NOT OK !!!   (error : %f)" %Lxx_err)

if Luu == 0:  print("Luu : OK    (error : %f)" %Luu_err)
else :     print("Luu : NOT OK !!!   (error : %f)" %Luu_err)

if Luu_noFri == 0:  print("Luu : OK    (error : %f) , no friction cone" %Luu_err_noFri)
else :     print("Luu : NOT OK !!!   (error : %f) , no friction cone" %Luu_err_noFri )



if Lx == 0 and Lu == 0 and Lxx == 0 and Luu_noFri == 0 and Fu == 0 and Fx == 0: print("\n      -->      Derivatives : OK")
else : print("\n         -->    Derivatives : NOT OK !!!")