# coding: utf8
import sys
import os 
from sys import argv
sys.path.insert(0, os.getcwd()) # adds current directory to python path

import numpy as np
import matplotlib.pylab as plt
from TSID_Debug_controller_four_legs_fb_vel import controller, dt
from crocoddyl_eval.test_6.main import run_scenario 
from IPython import embed
import Joystick

import multiprocessing as mp
import time

envID = 0
velID = 0

dt_mpc = 0.01 # Time step of the MPC
k_mpc = int(dt_mpc / dt)  # dt is dt_tsid, defined in the TSID controller script
t = 0.0  # Time
n_periods = 1  # Number of periods in the prediction horizon
T_gait = 0.32     # Duration of one gait period
N_SIMULATION = 10000  # number of simulated TSID time steps

# Which MPC solver you want to use
# True to have fstep planner, False to have optim dt
type_MPC = False

# Whether PyBullet feedback is enabled or not
pyb_feedback = True

#################
# RUN SCENARIOS #
#################

desired_speed = np.zeros(6)
desired_speed[0] = 0.0
gait , logger  = run_scenario(envID, velID, dt_mpc, k_mpc, t, n_periods, T_gait, N_SIMULATION, type_MPC, pyb_feedback , desired_speed)


# No Optim DT
if type_MPC == True : 
    pathIn = "crocoddyl_eval/test_6/log_eval/behaviour/"
# Optim DT
else : 
    pathIn = "crocoddyl_eval/test_6/log_eval/behaviour_dt/"

print("Saving logs...")

# np.save(pathIn +  "xref.npy" , xref )
# np.save(pathIn +  "l_feet_.npy" , l_feet_ )

# Optim DT
np.save(pathIn +  "gait.npy" , gait )
np.save(pathIn +  "lC.npy" , logger.oC )
np.save(pathIn +  "RPY.npy" , logger.RPY )
np.save(pathIn +  "lV.npy" , logger.oV )
np.save(pathIn +  "lW.npy" , logger.oW )

# np.save(pathIn +  "oC.npy" , logger.oC )
# np.save(pathIn +  "o_feet_.npy" , o_feet_ )
# np.save(pathIn +  "o_feet_heur.npy" , o_feet_heur )
# np.save(pathIn +  "pred_trajectories.npy" , pred_trajectories )
# np.save(pathIn +  "pred_forces.npy" , pred_forces )
# np.save(pathIn +  "feet_pos_target.npy" , logger.feet_pos_target )
# np.save(pathIn +  "lfeet_pos.npy" , logger.lfeet_pos )
# np.save(pathIn +  "lfeet_vel.npy" , logger.lfeet_vel )
# np.save(pathIn +  "lfeet_acc.npy" , logger.lfeet_acc )
# np.save(pathIn +  "pos_pred_local.npy" , pos_pred_local )
# np.save(pathIn +  "pos_pred_local_heur.npy" , pos_pred_local_heur )

# logger.plot_state()
# logger.plot_tracking_foot()
# plt.show(block=True)

quit()