Unit test to check the derivatives of the models. For most of the models, the hessian matrix can not be verified because the cost is not expressed as C = R^TR. 

- unittest_optim_period_dfeet.py : Check the derivatives of the model with time augmented state (nx = 21) and the delta feet command (nu = 4, 2 feet switch in X,Y plan). quadruped_walkgen.ActionModelQuadrupedStepTime()
