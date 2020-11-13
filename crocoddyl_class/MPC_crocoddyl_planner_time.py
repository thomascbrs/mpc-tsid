# coding: utf8
import sys
import os 
from sys import argv
sys.path.insert(0, os.getcwd()) # adds current directory to python path
import crocoddyl
import numpy as np
import quadruped_walkgen
import utils
import pinocchio as pin


class MPC_crocoddyl_planner_time():
    """Wrapper class for the MPC problem to call the ddp solver and 
    retrieve the results. 

    Args:
        dt (float): time step of the MPC
        T_mpc (float): Duration of the prediction horizon
        mu (float): Friction coefficient
        inner(bool): Inside or outside approximation of the friction cone
    """

    def __init__(self, dt = 0.02 , T_mpc = 0.32 ,  mu = 1, inner = True  , warm_start = False , min_fz = 0.0 , n_periods = 1):    

        # Time step of the solver
        self.dt = dt

        # Period of the MPC
        self.T_mpc = T_mpc

        # Number of period : not used yet
        self.n_periods = n_periods

        # Mass of the robot 
        self.mass = 2.50000279 

        # Inertia matrix of the robot in body frame 
        self.gI = np.array([[3.09249e-2, -8.00101e-7, 1.865287e-5],
                        [-8.00101e-7, 5.106100e-2, 1.245813e-4],
                        [1.865287e-5, 1.245813e-4, 6.939757e-2]])  

        # Friction coefficient
        if inner :
            self.mu = (1/np.sqrt(2))*mu
        else:
            self.mu = mu
        
        # Weights Vector : States
        self.w_x = 0.3
        self.w_y = 0.3
        self.w_z = 2
        self.w_roll = 0.9
        self.w_pitch = 1.
        self.w_yaw = 0.4
        self.w_vx =  1.5*np.sqrt(self.w_x)
        self.w_vy =  2*np.sqrt(self.w_y)
        self.w_vz =  1*np.sqrt(self.w_z)
        self.w_vroll =  0.05*np.sqrt(self.w_roll)
        self.w_vpitch =  0.07*np.sqrt(self.w_pitch)
        self.w_vyaw =  0.05*np.sqrt(self.w_yaw)
        self.stateWeights = np.array([self.w_x, self.w_y, self.w_z, self.w_roll, self.w_pitch, self.w_yaw,
                                    self.w_vx, self.w_vy, self.w_vz, self.w_vroll, self.w_vpitch, self.w_vyaw])

        # Weight Vector : Force Norm
        self.forceWeights = np.array(4*[0.01,0.01,0.01])

        # Weight Vector : Friction cone cost
        self.frictionWeights = 10.

        

        # Min/Max normal force (N)
        self.min_fz = min_fz
        self.max_fz = 25

        # Gait matrix
        self.gait = np.zeros((20, 5))
        self.gait_old = np.zeros((20, 5))
        # Index of the last line
        self.index = 0

        # Position of the feet in local frame
        self.fsteps = np.full((20, 13), 0.0)        

        # List of the actionModel
        self.ListAction = [] 

        # Warm start
        self.x_init = []
        self.u_init = []       

        # Weights on the shoulder term : term 1
        # self.shoulderWeights = np.array(4*[0.3,0.4])
        self.heuristicWeights = np.array(4*[0.0,0.0])
        # symmetry & centrifugal term in foot position heuristic
        self.centrifugal_term = True
        self.symmetry_term = True

        # Weight on the step command
        # self.stepWeights = np.array([0.1,0.3,0.1,0.3])   
        # self.stepWeights2 = np.array([0.2,0.4,0.2,0.4]) 
        self.stepWeights = np.array([0.4,0.4,0.4,0.4])   
        self.stepWeights2 =  np.array([0.4,0.4,0.4,0.4]) 

        # Weight on the step command for dt : 
        self.dt_ref = 0.02 # 0.5*dt_weight^2*||dt - dt_ref||^2
        self.dt_weight = 0.0

        N_nodes = self.T_mpc/self.dt * n_periods / 2
        self.T_gait_min = 0.24/2 #120ms for half period 
        self.T_gait_max = 2/2 #460ms for half period 
        self.dt_min = self.T_gait_min / N_nodes
        self.dt_max = self.T_gait_max / N_nodes
        self.dt_weight_bound = 0
        # print("dt_min : " , self.dt_min)
        # print("dt_max : " , self.dt_max)

        
        self.speed_weight = 0
        self.nb_nodes = self.T_mpc/self.dt * n_periods / 2 - 1 

        # Weights on the previous position predicted : term 2 
        self.lastPositionWeights = np.full(8,2.)

        # When the the foot reaches 10% of the flying phase, the optimisation of the foot 
        # positions stops by setting the "lastPositionWeight" on. 
        # For exemple, if T_mpc = 0.32s, dt = 0.02s, one flying phase period lasts 7 nodes. 
        # When there are 6 knots left before changing steps, the next knot will have its relative weight activated
        self.stop_optim = 0.1
        self.index_stop = int((1 - self.stop_optim)*(int(0.5*self.T_mpc/self.dt) - 1))     

        # Index of the control cycle to start the "stopping optimisation"
        self.start_stop_optim = 20

        # Preticted position of feet computed by previous cycle, it will be used with
        # the self.lastPositionWeights weight.
        # The world position of foot can be used with interface.oMl ...
        self.l_fsteps = np.zeros((3,4))   
        self.o_fsteps = np.zeros((3,4)) 

        # Max iteration ddp solver
        self.max_iteration = 100

        # Warm Start for the solver
        self.warm_start = warm_start  
        
        # Shooting problem
        self.problem = None
  
        # ddp solver
        self.ddp = None

        # Xs results without the actionStepModel
        self.Xs = np.zeros((21,int(T_mpc/dt)*n_periods))
        # self.Us = np.zeros((12,int(T_mpc/dt)))

        # Initial foot location (local frame, X,Y plan)
        self.p0 = [ 0.1946,0.15005, 0.1946,-0.15005, -0.1946,   0.15005 ,-0.1946,  -0.15005]

        self.shoulderPosition = np.array(self.p0)

        # Period list
        self.ListPeriod = []
        self.Max_nodes = self.T_mpc/self.dt * n_periods
        self.gait_new = np.zeros((20, 5))

        # Weights for dt model : 
        self.dt_weight_bound_cmd = 1000000. # Upper/lower bound
        self.dt_weight_cmd = 0. # ||U-dt||^2
        # States
        # Heuristic position

        self.relative_forces = True

        #Param for gait
        self.nb_nodes_horizon = 8
        self.T_min = 0.32
        self.T_max = 0.82
        self.node_init = self.T_min/(2*self.dt) - 1
        self.dt_init = self.dt
        self.term_factor = 5
        self.o_feet = np.zeros((3, 4))
        self.xref = None
        self.results_dt = np.zeros(2)   # dt1 and dt2 --dt0 always to 0.01

         # Weight on the shoulder term : 
        self.shoulderWeights = 0.1
        self.shoulder_hlim = 0.225 
        self.speed_weight2 = 5

        self.speed_weight_first = 1.
        self.feet_param = np.zeros((3,4))
        self.first_step = True
        # Weight for period optim
        self.vlim = 1.5
        # self.vlim = 3.6




    def solve(self, k, xref , interface , oMl = pin.SE3.Identity()):
        """ Solve the MPC problem 

        Args:
            k : Iteration 
            xref : the desired state vector
            l_feet : current position of the feet
        """ 


        # Update the dynamic depending on the predicted feet position
        self.updateProblem( k , xref , interface , oMl)

        # print(self.gait[:2,:])
        # print(interface.l_feet)
        # for elt in self.ddp.problem.runningModels : 
        #     print(elt.__class__.__name__)

        # Solve problem
        self.ddp.solve(self.x_init,self.u_init, 100) 
 
        print(self.gait[0,:])
        print(self.xref[6:9,0])
        for i in range(len(self.ddp.problem.runningModels)) : 
            if self.ddp.problem.runningModels[i].nu == 4 :  
                if self.ddp.problem.runningModels[i].first_step == True : 
                    self.ddp.problem.runningModels[i].speed_weight = self.speed_weight_first
                else : 
                    self.ddp.problem.runningModels[i].speed_weight = self.speed_weight2
                self.ddp.problem.runningModels[i].stepWeights = self.stepWeights
            if self.ddp.problem.runningModels[i].nu == 1 :  
                self.ddp.problem.runningModels[i].dt_weight_cmd = 0.  

        self.ddp.solve(self.ddp.xs,self.ddp.us,100, isFeasible=True)    
        
        

        # Get the results
        self.get_fsteps()

        return 0

    def updateProblem(self,k,xref , interface , oMl = pin.SE3.Identity()   ):
        """Update the dynamic of the model list according to the predicted position of the feet, 
        and the desired state. 
        self.o_feet = np.zeros((3, 4))  # position of feet in world frame

        Args:
        """
        self.oMl = oMl
        # position of foot predicted by previous gait cycle in world frame
        for i in range(4):
            self.l_fsteps[:,i] = self.oMl.inverse() * self.o_fsteps[:,i] 
    
        if k > 0:            
            # Move one step further in the gait 
            self.roll()     
            if np.sum(self.gait[0,1:]) == 4 : 
                # 4 contact --> need previous control cycle to know which foot was on the ground
                # On swing phase before --> initialised below shoulder
                self.p0 = np.repeat(np.array([1,1,1,1]) - self.gait_old[0,1:],2)*(self.shoulderPosition + np.concatenate(4*[xref[:2,int(self.gait[0,0]) ]] ) )
                # On the ground before -->  initialised with the current feet position
                self.p0 += np.repeat(self.gait_old[0,1:],2)*interface.l_feet[0:2,:].reshape(8, order = 'F')
            else : 
                # On swing phase before --> initialised below shoulder
                self.p0 = np.repeat(np.array([1,1,1,1]) - self.gait[0,1:],2)*(self.shoulderPosition + np.concatenate(4*[xref[:2,int(self.gait[0,0]) ]  ]) )
                # On the ground before -->  initialised with the current feet position
                self.p0 +=  np.repeat(self.gait[0,1:],2)*interface.l_feet[0:2,:].reshape(8, order = 'F')      
       
        else : 
            # Create gait matrix
            self.create_walking_trot()

            # Update initial state of the problem
            self.p0 = interface.l_feet[0:2,:].reshape(8, order = 'F')

        # create xref vector
        # Only velocities are used here, x,y,yaw free
       
        self.xref = np.zeros((12,int(np.sum(self.gait[:,0]) + 1)  ))
        self.xref[:,0] = xref[:,0] # initial state
        self.xref[2,1:] = xref[2,1]  # z ref
       

        # xref from fsteplanner object has not the same size
        # --> adaptation in velocities and z position
        self.xref[6:,1:] = np.tile(xref[6:,1] , (self.xref.shape[1] - 1,1)).transpose()

        self.create_List_model( interface)

        return 0       
        
    def get_latest_result(self):
        """Return the desired contact forces that have been computed by the last iteration of the MPC
        Args:
        """
        # print(self.gait)
        if self.ListAction[0].__class__.__name__ == "ActionModelQuadrupedStepTime" or self.ListAction[0].__class__.__name__ == "ActionModelQuadrupedTime":
            return np.repeat(self.gait[0,1:] , 3)*np.reshape(np.asarray(self.ddp.us[1])  , (12,))
        else :
            return np.repeat(self.gait[0,1:] , 3)*np.reshape(np.asarray(self.ddp.us[0])  , (12,))

    def update_model_augmented(self , model ,  optim_period = False):
        '''Set intern parameters for augmented model type
        '''
        # Model parameters
        model.dt = self.dt 
        model.mass = self.mass      
        model.gI = self.gI 
        model.mu = self.mu
        model.min_fz = self.min_fz
        model.max_fz = self.max_fz

        # Weights vectors
        model.forceWeights = self.forceWeights
        model.frictionWeights = self.frictionWeights   

        # Weight on feet position 
        # will be set when needed
        model.lastPositionWeights = np.full(8,0.0)
        model.symmetry_term = self.symmetry_term
        model.centrifugal_term = self.centrifugal_term
        model.dt_ref = self.dt_ref
        model.dt_weight = 0.
        model.dt_weight_bound = self.dt_weight_bound
        model.dt_min = self.dt_min
        model.dt_max = self.dt_max
        model.shoulderPosition = self.shoulderPosition

        model.relative_forces = self.relative_forces

        model.shoulderWeights = self.shoulderWeights
        model.shoulder_hlim = self.shoulder_hlim      

        if optim_period : 
            # model.heuristicWeights =  np.zeros(8)
            model.heuristicWeights  = self.heuristicWeights
            # model.stateWeights = np.concatenate([np.array([0.0, 0.0, ]), self.stateWeights[2:]])
            model.stateWeights =  self.stateWeights
        else : 
            model.heuristicWeights =  self.heuristicWeights
            model.stateWeights =  self.stateWeights

        return 0
    
    def update_model_step_feet(self , model , optim_period = True):
        """Set intern parameters for step model type
        """
        
        model.symmetry_term = self.symmetry_term
        model.centrifugal_term = self.centrifugal_term
       
        model.vlim = self.vlim
        model.nb_nodes = self.nb_nodes

        if optim_period : 
            model.stepWeights = self.stepWeights
            model.speed_weight = self.speed_weight
            model.heuristicWeights =  np.zeros(8)
            # model.stateWeights = np.array([0.0, 0.0, self.w_z, self.w_roll, self.w_pitch, self.w_yaw,
            #                         self.w_vx, self.w_vy, self.w_vz, self.w_vroll, self.w_vpitch, self.w_vyaw])
            model.stateWeights = np.zeros(12)
        else : 
            model.stepWeights = self.stepWeights
            model.speed_weight = 0.
            model.heuristicWeights =  self.heuristicWeights
            model.stateWeights =  self.stateWeights

        return 0
    
    def update_model_step_time(self , model , optim_period = True ):
        """Set intern parameters for step model type
        """        
        
        model.dt_ref = self.dt_ref
        model.symmetry_term = self.symmetry_term
        model.centrifugal_term = self.centrifugal_term
        model.T_gait = self.T_mpc

        model.dt_weight_bound_cmd = self.dt_weight_bound_cmd
        model.dt_weight_cmd = 0.
        model.dt_min = self.dt_min
        model.dt_max = self.dt_max

        if optim_period :
            model.heuristicWeights = np.zeros(8)
            model.stateWeights = np.zeros(12)
        else : 
            model.heuristicWeights =  self.heuristicWeights
            model.stateWeights = self.stateWeights

        return 0

    def create_List_model(self  , interface):
        """Create the List model using ActionQuadrupedModel()  
         The same model cannot be used [model]*(T_mpc/dt) because the dynamic changes for each nodes
        """

        # Prepare pos,vel,acc f flying feet for step feet model and trajectory 5D
        j = 0 
        while(np.sum(self.gait[j,1:]) != 4 ):
            j = j+1
        
        if j == 0 :
            gait_flying = self.gait[j,1:] - self.gait_old[0,1:]
        else : 
            gait_flying = self.gait[j,1:] - self.gait[j-1,1:]
        
        if gait_flying[0] == 1 : 
            # 1 00 1
            # pos speed acc px, 1
            self.feet_param[0,0] =  self.p0[0] - interface.l_feet[0,0]
            self.feet_param[1,0] = interface.lv_feet[0,0]
            self.feet_param[2,0] = interface.la_feet[0,0]
            # pos speed acc py, 1
            self.feet_param[0,1] = self.p0[1] - interface.l_feet[1,0]
            self.feet_param[1,1] = interface.lv_feet[1,0]
            self.feet_param[2,1] = interface.la_feet[1,0]

            self.feet_param[0,2] = self.p0[6] - interface.l_feet[0,3]
            self.feet_param[1,2] = interface.lv_feet[0,3]
            self.feet_param[2,2] = interface.la_feet[0,3]
            # pos speed acc py, 1
            self.feet_param[0,3] = self.p0[7] - interface.l_feet[1,3]
            self.feet_param[1,3] = interface.lv_feet[1,3]
            self.feet_param[2,3] = interface.la_feet[1,3]
        else : 
            #0 1 1 0
            self.feet_param[0,0] = self.p0[2] - interface.l_feet[0,1]
            self.feet_param[1,0] = interface.lv_feet[0,1]
            self.feet_param[2,0] = interface.la_feet[0,1]
            # pos speed acc py, 1
            self.feet_param[0,1] = self.p0[3] - interface.l_feet[1,1]
            self.feet_param[1,1] = interface.lv_feet[1,1]
            self.feet_param[2,1] = interface.la_feet[1,1]

            self.feet_param[0,2] = self.p0[4] - interface.l_feet[0,2]
            self.feet_param[1,2] = interface.lv_feet[0,2]
            self.feet_param[2,2] = interface.la_feet[0,2]
            # pos speed acc py, 1
            self.feet_param[0,3] = self.p0[5] - interface.l_feet[1,2]
            self.feet_param[1,3] = interface.lv_feet[1,2]
            self.feet_param[2,3] = interface.la_feet[1,2]
        
        # self.feet_param = np.zeros((3,4))

        j = 0
        k_cum = 0
        i = 0
        self.ListAction = []

        # WARM START
        x1 = np.zeros(21)
        x1[2] = 0.2027
        x1[-1] = self.dt_init
        u1 = np.array([0.1,0.1,8,0.1,0.1,8,0.1,0.1,8,0.1,0.1,8])
        self.x_init = []
        self.u_init = []
        # Iterate over all phases of the gait
        # The first column of xref correspond to the current state 
        while (self.gait[j, 0] != 0):
            for i in range(k_cum, k_cum+np.int(self.gait[j, 0])):

                if j == 0 and np.sum(self.gait[j, 1:]) == 2 and i == 1  :  
                    modelTime = quadruped_walkgen.ActionModelQuadrupedTime()
                    modelTime.updateModel(np.reshape(self.l_fsteps, (3, 4), order='F') , self.xref[:, i]  , self.gait[j, 1:]) 
                                    
                    # Update intern parameters
                    self.update_model_step_time(modelTime , True)
                    modelTime.dt_min = self.dt
                    # %of gait done 
                    modelTime.dt_max =  self.T_max/(2*(self.nb_nodes + 1)) 
                    modelTime.dt_weight_cmd = 1000
                    modelTime.dt_ref = modelTime.dt_min
            
                    self.ListAction.append(modelTime)   
                    self.x_init.append(x1)
                    self.u_init.append(np.array([self.dt_init]))

                elif j == 0 and np.sum(self.gait[j, 1:]) == 4 :

                    if np.sum(self.gait[0,1:] - self.gait_old[0,1:]) != 0 and i == 0 : 
                        model = quadruped_walkgen.ActionModelQuadrupedStepTime()
                        self.update_model_step_feet(model , True)                

                        model.updateModel(self.feet_param , self.xref[:, i]  ,  self.gait[j, 1:] - self.gait_old[j, 1:])
                        model.nb_nodes = self.gait[j,0]
                        model.first_step = self.first_step
                        # Update intern parameters
                        self.ListAction.append(model)
                        self.x_init.append(x1)
                        self.u_init.append(np.zeros(4))

                    if i == 1 : 
                        # 2 | 1 1 1 1 
                        modelTime = quadruped_walkgen.ActionModelQuadrupedTime()
                        modelTime.updateModel(np.reshape(self.l_fsteps, (3, 4), order='F') , self.xref[:, i]  , self.gait[j, 1:]) 
                                        
                        # Update intern parameters
                        self.update_model_step_time(modelTime , True)
                        modelTime.dt_min = self.dt
                        modelTime.dt_max =  self.T_max/(2*(self.nb_nodes + 1)) 
                        modelTime.dt_weight_cmd = 1000
                        modelTime.dt_ref = modelTime.dt_min
                        self.ListAction.append(modelTime)   
                        self.x_init.append(x1)
                        self.u_init.append(np.array([self.dt_init]))

                    

                

                model = quadruped_walkgen.ActionModelQuadrupedAugmentedTime()
                self.update_model_augmented(model ,True)

                model.updateModel(np.reshape(self.l_fsteps, (3, 4), order='F') , self.xref[:, i+1]  , self.gait[j, 1:])
            
                # Update intern parameters
                self.ListAction.append(model)

                self.x_init.append(x1)
                self.u_init.append(np.repeat(self.gait[j,1:] , 3)*u1)
            
            if np.sum(self.gait[j+1, 1:]) == 4 : # No optimisation on the first line                   
            
                model = quadruped_walkgen.ActionModelQuadrupedStepTime()
                self.update_model_step_feet(model , True)
            

                model.updateModel(self.feet_param , self.xref[:, i+1]  ,  self.gait[j+1, 1:] - self.gait[j, 1:])
                model.nb_nodes = self.gait[j,0]
                model.vlim = self.vlim
                if j == 1 : 
                    model.first_step = self.first_step
                else : 
                    model.first_step = False
                # Update intern parameters
                self.ListAction.append(model)
                self.x_init.append(x1)
                self.u_init.append(np.zeros(4))

                if j == 0 and np.sum(self.gait[0,0]) == 1 : 
                        # 1 | 1 0 0 1
                        # 1 | 1 1 1 1 
                        modelTime = quadruped_walkgen.ActionModelQuadrupedTime()
                        modelTime.updateModel(np.reshape(self.l_fsteps, (3, 4), order='F') , self.xref[:, i]  , self.gait[j, 1:]) 
                                        
                        # Update intern parameters
                        self.update_model_step_time(modelTime , True)
                        modelTime.dt_min = self.dt
                        modelTime.dt_max =  self.T_max/(2*(self.nb_nodes + 1)) 
                        modelTime.dt_weight_cmd = 1000
                        modelTime.dt_ref = modelTime.dt_min
                        self.ListAction.append(modelTime)   
                        self.x_init.append(x1)
                        self.u_init.append(np.array([self.dt_init])) 
                    
            if np.sum(self.gait[j+1, 1:]) == 2 :    

                modelTime = quadruped_walkgen.ActionModelQuadrupedTime()
                
                    
                modelTime.updateModel(np.reshape(self.l_fsteps, (3, 4), order='F') , self.xref[:, i+1]  ,self.gait[j, 1:]) 
                
                # Update intern parameters
                self.update_model_step_time(modelTime , True)
                if  j== 1 or j == 0 :
                    modelTime.dt_weight_cmd = 1000
                    modelTime.dt_min = self.T_min/(2*(self.nb_nodes_horizon + 1))
                    modelTime.dt_max = self.T_max/(2*(self.nb_nodes_horizon + 1)) 
                    modelTime.dt_ref = modelTime.dt_min
                    self.u_init.append(np.array([self.results_dt[0]]))
                if  j== 3 or j ==2 :
                    modelTime.dt_weight_cmd = 1000
                    modelTime.dt_min = self.T_min/(2*(self.nb_nodes_horizon + 1))
                    modelTime.dt_max = self.T_max/(2*(self.nb_nodes_horizon + 1)) 
                    self.u_init.append(np.array([self.results_dt[1]]))
                    modelTime.dt_ref = modelTime.dt_min
                self.ListAction.append(modelTime)   
                self.x_init.append(x1)
                

            k_cum += np.int(self.gait[j, 0])
            j += 1


        # Model parameters of terminal node  
        self.terminalModel = quadruped_walkgen.ActionModelQuadrupedAugmentedTime()
        self.terminalModel.updateModel(np.reshape(self.l_fsteps, (3, 4), order='F') , self.xref[:, -1]  , self.gait[j-1, 1:]) 
        self.update_model_augmented(self.terminalModel , True)
        self.x_init.append(np.zeros(21))
        # Weights vectors of terminal node
        self.terminalModel.forceWeights = np.zeros(12)
        self.terminalModel.frictionWeights = 0.
        self.terminalModel.heuristicWeights = np.full(8,0.0)
        self.terminalModel.lastPositionWeights =  np.full(8,0.0)
        self.terminalModel.stateWeights = self.term_factor*self.terminalModel.stateWeights 

        # Shooting problem
        self.problem = crocoddyl.ShootingProblem(np.zeros(21),  self.ListAction, self.terminalModel)
        self.problem.x0 = np.concatenate([self.xref[:,0] , self.p0 , [self.dt_init]   ])
        self.ddp = crocoddyl.SolverDDP(self.problem)

       

    
        return 0

    def create_walking_trot(self , initial_pos = np.array([1,0,0,1])):
        """Create the matrices used to handle the gait and initialize them to perform a walking trot

        self.gait and self.fsteps matrices contains information about the walking trot
        """

        # Old matrix --> usefull to compute p0
        self.gait_old = self.gait

        # Number of timesteps in a half period of gait
        N = np.int(self.T_mpc/(2*self.dt)) - 1

        # Starting status of the gait
        # 4-stance phase, 2-stance phase, 4-stance phase, 2-stance phase
        self.gait = np.zeros((self.fsteps.shape[0], 5))

        self.gait[0,0] = self.node_init
        self.gait[0,1:] = initial_pos
        self.gait[1,:] = np.array([1,1,1,1,1])
        self.gait[2,0] = self.nb_nodes_horizon
        self.gait[2,1:] = np.full(4,1) - initial_pos
        self.gait[3,:] = np.array([1,1,1,1,1])
        self.gait[4,0] = self.nb_nodes_horizon
        self.gait[4,1:] = initial_pos
        self.gait[5,:] = np.array([1,1,1,1,1])


        return 0

    def roll(self):
        """Move one step further in the gait cycle

        Decrease by 1 the number of remaining step for the current phase of the gait and increase
        by 1 the number of remaining step for the last phase of the gait (periodic motion)

        Add and remove corresponding model in ListAction
        """
        
        # Index of the first empty line
        # index = next((idx for idx, val in np.ndenumerate(self.gait[:, 0]) if val==0.0), 0.0)[0]
        self.gait_old = self.gait

        if int(self.gait[0,0]) > 1 : # no need switch
            self.gait[0,0] -= 1 
        else : 
            if int(np.sum(self.gait[0,1:])) == 2. : # DS, switch to 4S
                self.gait[0,:] = np.full(5,0)
                self.gait = np.roll(self.gait, -1, axis=0)
            else : # 4S  --> restart with 
                self.gait[0,0] = self.node_init
                self.gait[0,1:] = self.gait[1,1:]
                self.gait[1,:] = np.array([1,1,1,1,1])
                self.gait[2,0] = self.nb_nodes_horizon
                self.gait[2,1:] = np.full(4,1) - self.gait[0,1:]
                self.gait[3,:] = np.array([1,1,1,1,1])
                self.gait[4,0] = self.nb_nodes_horizon
                self.gait[4,1:] = self.gait[0,1:]
                self.gait[5,:] = np.array([1,1,1,1,1])
              
        return 0

    def get_fsteps(self):
        """Create the matrices fstep, the position of the feet predicted during the control cycle.

        To be used after the solve function.
        """
        ##################################################
        # Get command vector without actionModelStep node
        ##################################################

        Us = self.ddp.us
        Liste = [x for x in Us if (x.size != 4 and x.size != 1) ]
        self.Us =  np.array(Liste)[:,:].transpose()

        ################################################
        # Get state vector without actionModelStep node
        ################################################

        Xs = [self.ddp.xs[i] for i in range(len(self.ddp.us)) if (self.ddp.us[i].size != 4 and self.ddp.us[i].size != 1) ]
        Xs.append(self.ddp.xs[-1]) #terminal node
        self.Xs = np.array(Xs).transpose()

        # Dt optimised
        results_dt = [self.ddp.xs[i+1][-1] for i in range(len(self.ddp.us)) if self.ddp.us[i].size == 1 ]

        print(results_dt)
        
        
        # print(self.gait)
        # for elt in self.ddp.problem.runningModels : 
        #     print(elt.__class__.__name__)

        if len(results_dt) == 2 : 
            self.results_dt[0] = results_dt[0]
            self.results_dt[1] = results_dt[1]
        else : 
            self.results_dt[0] = results_dt[1]
            self.results_dt[1] = results_dt[2]

      
        ########################################
        # Compute fheuristicWeightssteps using the state vector
        ################################16 ########

        j = 0
        k_cum = 0

        # self.fsteps[0,0] = self.gait[0,0]

        self.ListPeriod = []
        self.gait_new = np.zeros((20,5))

        # Iterate over all phases of the gait
        while (self.gait[j, 0] != 0):
            

            self.fsteps[j ,1: ] = np.repeat(self.gait[j,1:] , 3)*np.concatenate([self.Xs[12:14 , k_cum ],[0.],self.Xs[14:16 , k_cum ],[0.],
                                                                                self.Xs[16:18 , k_cum ],[0.],self.Xs[18:20 , k_cum ],[0.]])    
        
            # if int(self.gait[0,0]) > 1 :
            #     if (j == 0 and int(np.sum(self.gait[0,1:])) == 2) or (j==1 and int(np.sum(self.gait[0,1:])) == 4) : 
            #         self.fsteps[j,0] = np.around(1 + ((self.gait[j,0] - 1)*self.Xs[20,k_cum] / self.dt)  ,decimals = 0)
            #         self.gait_new[j,0] =  np.around((self.gait[j,0]*self.Xs[20,k_cum] / self.dt)  ,decimals = 0)
            #     else : 
            #         self.fsteps[j,0] = self.gait[j,0]
            #         self.gait_new[j,0] = self.gait[j,0]

            # else : 
            #     self.fsteps[j,0] = self.gait[j,0]
            #     self.gait_new[j,0] = self.gait[j,0]

            # if j == 0 and int(np.sum(self.gait[0,1:])) == 2 :   # 1 | 1 0 0 1  
            if int(self.gait[0,0]) > 1 : 
                if j == 0 :
                    # k_cum + 1 --> 1st is at dt, optim after the first node
                    # 2 | 1 1 1 1    or 2 | 1 0 0 1 
                    self.fsteps[j,0] = np.around(1 + ((self.gait[j,0] - 1)*self.Xs[20,k_cum + 1] / self.dt)  ,decimals = 0)
                    self.gait_new[j,0] =  np.around(1 + ((self.gait[j,0] - 1)*self.Xs[20,k_cum + 1] / self.dt)  ,decimals = 0)
                elif j == 1 and int(np.sum(self.gait[1,1:])) == 4 :
                    # No 1st node
                    # self.fsteps[j,0] = np.around((self.gait[j,0]*self.Xs[20,k_cum] / self.dt)  ,decimals = 0)
                    # self.gait_new[j,0] =  np.around((self.gait[j,0]*self.Xs[20,k_cum] / self.dt)  ,decimals = 0)
                    self.fsteps[j,0] = self.gait[j,0]
                    self.gait_new[j,0] = self.gait[j,0]
                else : 
                    self.fsteps[j,0] = self.gait[j,0]
                    self.gait_new[j,0] = self.gait[j,0]
            else : 
                if j == 1 and np.sum(self.gait[1,1:]) == 4 : 
                    # self.fsteps[j,0] = self.gait[j,0]
                    # self.gait_new[j,0] = self.gait[j,0]
                    self.fsteps[j,0] = np.around((self.gait[j,0]*self.Xs[20,k_cum] / self.dt)  ,decimals = 0)
                    self.gait_new[j,0] =  np.around((self.gait[j,0]*self.Xs[20,k_cum] / self.dt)  ,decimals = 0)
                else : 
                    self.fsteps[j,0] = self.gait[j,0]
                    self.gait_new[j,0] = self.gait[j,0]


                 
                
            self.gait_new[j,1:] = self.gait[j,1:]
            
            
            k_cum += np.int(self.gait[j, 0])
            j += 1    
        
        if int(self.gait[0,0] - self.gait_new[0,0]) != 0 : 
            print("\n")
            print("         SLOWWWW       ")
            print("\n")
        # self.gait_old = self.gait
        self.gait = self.gait_new    


        ####################################################
        # Compute the current position of feet in contact
        # and the position of desired feet in flying phase
        # in local frame
        #####################################################

        for i in range(4):
            index = next((idx for idx, val in np.ndenumerate(self.fsteps[:, 3*i+1]) if ((not (val==0)) and (not np.isnan(val)))), [-1])[0]
            #print(str(i) + ": ", (np.array([fsteps[index, (1+1+i*3):(3+i*3)]]).ravel()))
            # pos_tmp = np.reshape(np.array(self.oMl * (np.array([self.fsteps[index, (1+i*3):(4+i*3)]]).transpose())) , (3,1) )
            pos_tmp = np.reshape( np.array([self.fsteps[index, (1+i*3):(4+i*3)]]).transpose() , (3,1) )   
            
            self.l_fsteps[:2,i] = pos_tmp[0:2, 0]
            pos_tmp = np.reshape(np.array(self.oMl * (np.array([self.fsteps[index, (1+i*3):(4+i*3)]]).transpose())) , (3,1) )
            self.o_fsteps[:2, i] = pos_tmp[0:2, 0]

        return self.fsteps

    def updatePositionWeights(self) : 

        """Update the parameters in the ListAction to keep the next foot position at the same position computed by the 
         previous control cycle and avoid re-optimization at the end of the flying phase
        """

        if self.gait[0,0] == self.index_stop : 
             self.ListAction[int(self.gait[0,0])+ 1].lastPositionWeights =  np.repeat((np.array([1,1,1,1]) - self.gait[0,1:]) , 2 )*  self.lastPositionWeights
       
        return 0

    def get_xrobot(self):
        """Returns the state vectors predicted by the mpc throughout the time horizon, the initial column is deleted as it corresponds
        initial state vector
        Args:
        """

        return np.array(self.ddp.xs)[1:,:].transpose()

    def get_fpredicted(self):
        """Returns the force vectors command predicted by the mpc throughout the time horizon, 
        Args:
        """

        return np.array(self.ddp.us)[:,:].transpose()[:,:]




        

    

