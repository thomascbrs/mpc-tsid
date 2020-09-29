# coding: utf8
import sys
import os 
from sys import argv
sys.path.insert(0, os.getcwd()) # adds current directory to python path

import numpy as np
import matplotlib.pylab as plt
from TSID_Debug_controller_four_legs_fb_vel import controller, dt
from crocoddyl_eval.test_5.main import run_scenario 
from IPython import embed
import Joystick

import multiprocessing as mp
import time

envID = 0
velID = 0

dt_mpc = 0.02  # Time step of the MPC
k_mpc = int(dt_mpc / dt)  # dt is dt_tsid, defined in the TSID controller script
t = 0.0  # Time
n_periods = 1  # Number of periods in the prediction horizon
T_gait = 0.64  # Duration of one gait period
N_SIMULATION = 1000  # number of simulated TSID time steps

# Which MPC solver you want to use
# True to have PA's MPC, to False to have Thomas's MPC
type_MPC = False 

# Whether PyBullet feedback is enabled or not
pyb_feedback = True

#################
# RUN SCENARIOS #
#################

def run_simu(speed) : 
    desired_speed = np.zeros(6)
    desired_speed[0] = speed[0]
    desired_speed[1] = speed[1]
    desired_speed[5] = speed[5]

    return run_scenario(envID, velID, dt_mpc, k_mpc, t, n_periods, T_gait, N_SIMULATION, type_MPC, pyb_feedback , desired_speed)


V = [0.71428571,  0.0 , 0. , 0. , 0. ,0.19285714]
o_feet_ , o_feet_heur , gait_ , pred_trajectories  , l_feet_ , pos_pred_local , pos_pred_local_heur , logger = run_simu(V)

pathIn = "crocoddyl_eval/test_5/log_eval/test_2/"

print("Saving logs...")

# np.save(pathIn +  "oC.npy" , logger.oC )
# np.save(pathIn +  "o_feet_.npy" , o_feet_ )
# np.save(pathIn +  "l_feet_.npy" , l_feet_ )
# np.save(pathIn +  "o_feet_heur.npy" , o_feet_heur )
# np.save(pathIn +  "gait_.npy" , gait_ )
# np.save(pathIn +  "pred_trajectories.npy" , pred_trajectories )
# np.save(pathIn +  "feet_pos.npy" , logger.feet_pos )
# np.save(pathIn +  "feet_pos_target.npy" , logger.feet_pos_target )
# np.save(pathIn +  "pos_pred_local.npy" , pos_pred_local )
# np.save(pathIn +  "pos_pred_local_heur.npy" , pos_pred_local_heur )

logger.plot_state()
logger.plot_footsteps()
logger.plot_state()
logger.plot_tracking_foot()
plt.show(block=True)

quit()