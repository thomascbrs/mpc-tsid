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

N_trial = 50
a = 1 
b = -1


epsilon = 1e-3

################################################
## CHECK DERIVATIVE WITH NUM_DIFF , 
#################################################

actionModel = quadruped_walkgen.ActionModelQuadrupedStepTime()
actionModel.heuristicWeights =  np.zeros(8) # weight on the heuristic optim of the feet
actionModel.stateWeights = np.zeros(12) # State weight
actionModel.vlim = 0.12415
actionModel.nb_nodes = 15
actionModel.speed_weight = 15
actionModel.stepWeights = np.zeros(4)
actionModel.first_step = True
data = actionModel.createData()

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
  Lxu_err = 0
  Lxu = 0 

  for k in range(N_trial):    

    x = a + (b-a)*np.random.rand(21)
    x[-1] = np.random.rand(1)[0] #dt > 0 
    # x[-1] = 1
    u = a + (b-a)*np.random.rand(4)
    # u = np.array([2,2,2,2])

    # l_feet = np.random.rand(3,4)
    l_feet = np.zeros((3,4))
    # l_feet[0,1] = -1
    xref = np.random.rand(12)
    nb = np.random.rand(1)[0] 
    # nb = 0.8
    # Only 2 feet switch at the same time
    if nb > 0.5 :
      S = np.array([0,1,1,0])
    else : 
      S = np.array([1,0,0,1])

    # if k%2 == 0 : 
    #   actionModel.first_step = True
    # else : 
    #   actionModel.first_step = False
    actionModel.updateModel(l_feet , xref , S )
    # model_diff = crocoddyl.ActionModelNumDiff(actionModel)
    model_diff = quadruped_walkgen.ActionModelQuadrupedStepTime()
    model_diff.heuristicWeights =  np.zeros(8) # weight on the heuristic optim of the feet
    model_diff.stateWeights = np.zeros(12) # State weight
    model_diff.vlim = actionModel.vlim
    model_diff.nb_nodes = actionModel.nb_nodes
    model_diff.speed_weight = actionModel.speed_weight
    model_diff.stepWeights = np.zeros(4)
    model_diff.first_step = False
    model_diff.updateModel(l_feet , xref , S )
    dataCpp = model_diff.createData()
    
   
    # Run calc & calcDiff function : numDiff 
    
    model_diff.calc(dataCpp , x , u )
    model_diff.calcDiff(dataCpp , x , u )
    
    # Run calc & calcDiff function : c++
    actionModel.calc(data , x , u )
    actionModel.calcDiff(data , x , u )

    Lx +=  np.sum( abs((64/225*data.Lx - dataCpp.Lx )) >= epsilon  ) 
    Lx_err += np.sum( abs((64/225*data.Lx - dataCpp.Lx )) )  


    Lu +=  np.sum( abs((64/225*data.Lu - dataCpp.Lu )) >= epsilon  ) 
    Lu_err += np.sum( abs((64/225*data.Lu - dataCpp.Lu )) )  

    Lxx +=  np.sum( abs((64/225*data.Lxx - dataCpp.Lxx )) >= epsilon  ) 
    Lxx_err += np.sum( abs((64/225*data.Lxx - dataCpp.Lxx )) )  

    Luu +=  np.sum( abs((64/225*data.Luu - dataCpp.Luu )) >= epsilon  ) 
    Luu_err += np.sum( abs((64/225*data.Luu - dataCpp.Luu )) )  

    Fx +=  np.sum( abs((data.Fx - dataCpp.Fx )) >= epsilon  ) 
    Fx_err += np.sum( abs((data.Fx - dataCpp.Fx )) )  

    Fu +=  np.sum( abs((data.Fu - dataCpp.Fu )) >= epsilon  ) 
    Fu_err += np.sum( abs((data.Fu - dataCpp.Fu )) )  

    Lxu +=  np.sum( abs((64/225*data.Lxu - dataCpp.Lxu )) >= epsilon  ) 
    Lxu_err += np.sum( abs((64/225*data.Lxu - dataCpp.Lxu )) )  

   
    print(data.cost)
    print(dataCpp.cost)
    print("\n")
    print("Lu")
    print(data.Lu)
    print(dataCpp.Lu)
    print("\n")
    print("Lx")
    print(data.Lx)
    print(dataCpp.Lx)
    print("\n")
    print("Luu")
    print(data.Luu)
    print(dataCpp.Luu)
    print("\n")
    print("Lxx")
    print(data.Lxx[20,20])
    print(dataCpp.Lxx[20,20])





  
  Lx_err = Lx_err /N_trial
  Lu_err = Lu_err/N_trial
  Lxx_err = Lxx_err/N_trial    
  Luu_err = Luu_err/N_trial
  Fx_err = Fx_err/N_trial
  Fu_err = Fu_err/N_trial
  Lxu_err = Lxu_err/N_trial
  
  return Lx , Lx_err , Lu , Lu_err , Lxx , Lxx_err , Luu , Luu_err,   Fx, Fx_err, Fu , Fu_err , Lxu , Lxu_err


Lx , Lx_err , Lu , Lu_err , Lxx , Lxx_err , Luu , Luu_err ,  Fx, Fx_err, Fu , Fu_err , Lxu , Lxu_err = run_calcDiff_numDiff(epsilon)

print("\n \n")
print(" Action model class, step for position of foot, with time augmented state vector (nx = 21 , nu = 4) ")
print("\n ------------------------------------------ " )
print(" Checking implementation of the derivatives ")
print(" Using crocoddyl NumDiff class")
print(" ------------------------------------------ " )

print("\n")
print("Espilon : %f" %epsilon)
print("N_trial : %f" %N_trial)
print("\n")
print("State Weights : " , actionModel.stateWeights)
print("Heuristic Weights : " , actionModel.heuristicWeights)
print("Speed weight : " , actionModel.speed_weight)
print("stepWeight : " , actionModel.stepWeights)
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

if Lxu == 0:  print("Lxu : OK    (error : %f)" %Lxu_err)
else :     print("Lxu : NOT OK !!!   (error : %f)" %Lxu_err)


if Lx == 0 and Lu == 0 and Lxx == 0 and Fu == 0 and Fx == 0: print("\n      -->      Derivatives : OK")
else : print("\n         -->    Derivatives : NOT OK !!!")