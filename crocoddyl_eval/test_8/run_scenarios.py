# coding: utf8
import sys
import os 
from sys import argv
sys.path.insert(0, os.getcwd()) # adds current directory to python path

import numpy as np
from TSID_Debug_controller_four_legs_fb_vel import controller, dt
from crocoddyl_eval.test_8.main import run_scenario
from IPython import embed

envID = 0
velID = 2

dt_mpc = 0.01  # Time step of the MPC
k_mpc = int(dt_mpc / dt)  # dt is dt_tsid, defined in the TSID controller script
t = 0.0  # Time
n_periods = 1  # Number of periods in the prediction horizon
T_gait = 0.3      # Duration of one gait period
N_SIMULATION = 20000  # number of simulated TSID time steps

# Which MPC solver you want to use
# True to have PA's MPC, to False to have Thomas's MPC
type_MPC = False


# 0 --> Linear
# 1 --> Non Linear
# 2 --> Footstep planner
# 3 --> Footstep planner + optim
mpc_model = 2

# Whether PyBullet feedback is enabled or not
pyb_feedback = True

# Desired speed
# increasing by 0.1m.s-1 each second, and then 10s of simulation

#################
# RUN SCENARIOS #
#################

# desired_speed = [0.75,0.0,0.,0.,0.,0.]
desired_speed = [0.0,0.0,0.,0.,0.,0.]

# Run a scenario and retrieve data thanks to the logger
torques_ff , torques_pd , torques_sent, qtsid , vtsid = run_scenario(envID, velID, dt_mpc, k_mpc, t, n_periods, T_gait, N_SIMULATION, type_MPC, pyb_feedback,mpc_model ,desired_speed)

#################
# RECORD LOGGERS
#################

pathIn = "crocoddyl_eval/test_8/log_eval/"

print("Saving logs...")

np.save(pathIn +  "torques_ff.npy" , torques_ff )
np.save(pathIn +  "torques_pd.npy" , torques_pd )

np.save(pathIn +  "torques_sent.npy" , torques_sent )
np.save(pathIn +  "qtsid.npy" , qtsid )
np.save(pathIn +  "vtsid.npy" , vtsid )



import matplotlib.pylab as plt
logger_ddp.plot_state()
logger_ddp.plot_footsteps()
plt.show(block=True)

quit()