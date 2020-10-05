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
        # self.gI = np.diag([0.00578574, 0.01938108, 0.02476124])
        self.gI = np.array([[3.09249e-2, -8.00101e-7, 1.865287e-5],
                        [-8.00101e-7, 5.106100e-2, 1.245813e-4],
                        [1.865287e-5, 1.245813e-4, 6.939757e-2]])  

        # Friction coefficient
        if inner :
            self.mu = (1/np.sqrt(2))*mu
        else:
            self.mu = mu
        
        # Weights Vector : States
        # self.stateWeights = np.array([1,1,150,35,30,8,20,20,15,4,4,8])
        # Weights Vector : States
        self.w_x = 0.3
        self.w_y = 0.3
        self.w_z = 2
        self.w_roll = 0.9
        self.w_pitch = 1.
        self.w_yaw = 0.4
        # self.w_vx =  1.5*np.sqrt(self.w_x)
        self.w_vx =  1.5*np.sqrt(0.3)
        self.w_vy =  2*np.sqrt(self.w_y)
        self.w_vz =  1*np.sqrt(self.w_z)
        self.w_vroll =  0.05*np.sqrt(self.w_roll)
        self.w_vpitch =  0.07*np.sqrt(self.w_pitch)
        self.w_vyaw =  0.05*np.sqrt(self.w_yaw)
        self.stateWeights = np.array([self.w_x, self.w_y, self.w_z, self.w_roll, self.w_pitch, self.w_yaw,
                                    self.w_vx, self.w_vy, self.w_vz, self.w_vroll, self.w_vpitch, self.w_vyaw])

        # Weight Vector : Force Norm
        # self.forceWeights = np.full(12,0.02)
        self.forceWeights = np.array(4*[0.01,0.01,0.01])

        # Weight Vector : Friction cone cost
        # self.frictionWeights = 10
        self.frictionWeights = 0.5

        # Max iteration ddp solver
        self.max_iteration = 50

        # Warm Start for the solver
        self.warm_start = warm_start

        # Minimum normal force (N)
        self.min_fz = min_fz
        self.max_fz = 25

        # Gait matrix
        self.gait = np.zeros((20, 5))
        self.gait_old = np.zeros((20, 5))
        self.index = 0

        # Position of the feet in local frame
        self.fsteps = np.full((20, 13), 0.0)        

        # List of the actionModel
        self.ListAction = [] 

        # Warm start
        self.x_init = []
        self.u_init = []       

        # Weights on the shoulder term : term 1
        self.shoulderWeights = np.array(4*[0.3,0.4])
        # Weights on the shoulder term : term 1
        self.heuristicWeights = np.array(4*[0.3,0.4])

        # symmetry & centrifugal term in foot position heuristic
        self.centrifugal_term = True
        self.symmetry_term = True

        # Weight on the step command
        self.stepWeights = np.full(4,0.8)            

        # Weight on the step command for dt : 
        self.dtStepWeight = 2   # 0.5*dtStepWeight^2*||u||^2
        self.dt_ref = 0.02 # 0.5*dt_weight^2*||dt - dt_ref||^2
        self.dt_initial = 0.02
        self.dt_weight = 0.0

        N_nodes = self.T_mpc/self.dt * n_periods / 2
        self.T_gait_min = 0.24/2 #120ms for half period 
        self.T_gait_max = 2/2 #460ms for half period 
        self.dt_min = self.T_gait_min / N_nodes
        self.dt_max = self.T_gait_max / N_nodes
        self.dt_weight_bound = 50000
        print("dt_min : " , self.dt_min)
        print("dt_max : " , self.dt_max)

        # Weight for period optim
        self.vlim = 2.
        self.speed_weight = 1000
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
        # self.o_fsteps = np.zeros((3,4))   
        
        # Shooting problem
        self.problem = None
  
        # ddp solver
        self.ddp = None

        # Xs results without the actionStepModel
        self.Xs = np.zeros((21,int(T_mpc/dt)*n_periods))
        # self.Us = np.zeros((12,int(T_mpc/dt)))

        # Initial foot location (local frame, X,Y plan)
        self.p0 = [ 0.1946,0.15005, 0.1946,-0.15005, -0.1946,   0.15005 ,-0.1946,  -0.15005]

        # Period list
        self.ListPeriod = []
        self.Max_nodes = self.T_mpc/self.dt * n_periods
        self.gait_new = np.zeros((20, 5))

        # Weights for dt model : 
        self.dt_weight_bound_cmd = 10000. # Upper/lower bound
        self.dt_weight_cmd = 0. # ||U-dt||^2
        # States
        # Heuristic position

        self.relative_forces = False

        


    def solve(self, k, xref , l_feet):
        """ Solve the MPC problem 

        Args:
            k : Iteration 
            xref : the desired state vector
            l_feet : current position of the feet
        """ 

        # Update the dynamic depending on the predicted feet position
        self.updateProblem( k , xref , l_feet )

        # Solve problem
        self.ddp.solve(self.x_init,self.u_init, self.max_iteration)        
        
        # Get the results
        self.get_fsteps()

        return 0

    def updateProblem(self,k,xref , l_feet ):
        """Update the dynamic of the model list according to the predicted position of the feet, 
        and the desired state. 

        Args:
        """
        # self.oMl = mpc_interface.oMl
        # position of foot predicted by previous gait cycle in world frame
        # for i in range(4):
        #     self.l_fsteps[:,i] = self.oMl.inverse() * self.o_fsteps[:,i] 
    
        # if k > 0:            
        #     # Move one step further in the gait 
        #     # Add and remove step model in the list of model
        #     self.roll() 
        #     self.create_List_model()

        #     # Update initial state of the problem
            
        #     if np.sum(self.gait[0,1:]) == 4 : 
        #         # 4 contact --> need previous control cycle to know which foot was on the ground
        #         # On swing phase before --> initialised below shoulder
        #         # p0 = np.repeat(np.array([1,1,1,1])-self.gait_old[0,1:],2)*self.p0  
        #         # # On the ground before -->  initialised with the current feet position
        #         # p0 +=  np.repeat(self.gait_old[0,1:],2)*l_feet[0:2,:].reshape(8, order = 'F')
        #         p0 = l_feet[0:2,:].reshape(8, order = 'F')
        #     else : 
                
        #         # On swing phase before --> initialised below shoulder
        #         # p0 = np.repeat(np.array([1,1,1,1])-self.gait[0,1:],2)*self.p0  
        #         # # On the ground before -->  initialised with the current feet position
        #         # p0 +=  np.repeat(self.gait[0,1:],2)*l_feet[0:2,:].reshape(8, order = 'F')
        #         p0 = l_feet[0:2,:].reshape(8, order = 'F')
       
        # else : 
        #     # Create gait matrix
        #     self.create_walking_trot()
        #     self.gait_old = self.gait    
        #     # According to the current footstepplanner, the walk start on the next phase
        #     self.roll()
        #     # First step : create the list of model
        #     self.create_List_model()
        #     # Update initial state of the problem with the shoulder position 
        #     p0 = self.p0

        p0 = np.repeat(np.array([1,1,1,1])-self.gait[0,1:],2)*self.p0  
        # On the ground before -->  initialised with the current feet position
        p0 +=  np.repeat(self.gait[0,1:],2)*l_feet[0:2,:].reshape(8, order = 'F')
        
        j = 0
        k_cum = 0
        L = []
        # Iterate over all phases of the gait
        # The first column of xref correspond to the current state 
        # Gap introduced to take into account the Step model (more nodes than gait phases )
        self.x_init = []
        self.u_init = []
        gap = 1

        while (self.gait[j, 0] != 0):
            
            for i in range(k_cum, k_cum+np.int(self.gait[j, 0])):

                if self.ListAction[i].__class__.__name__ == "ActionModelQuadrupedStepTime" :
                    self.x_init.append(np.zeros(21))
                    self.u_init.append(np.zeros(4))
                    if i == 0 : 
                        self.ListAction[i].updateModel(np.reshape(self.l_fsteps, (3, 4), order='F') , xref[:, i+gap]  , self.gait[0, 1:] - self.gait_old[0, 1:])
                        self.ListAction[i+1].updateModel(np.reshape(self.l_fsteps, (3, 4), order='F') , xref[:, i+gap]  , self.gait[j, 1:])
                        k_cum +=  1
                        gap -= 1                       
                        
                    else : 
                        self.ListAction[i].updateModel(np.reshape(self.l_fsteps, (3, 4), order='F') , xref[:, i+gap]  , self.gait[j, 1:] - self.gait[j-1, 1:])
                        self.ListAction[i+1].updateModel(np.reshape(self.l_fsteps, (3, 4), order='F') , xref[:, i+gap]  , self.gait[j, 1:])
                        k_cum +=  1
                        gap -= 1

                    self.x_init.append(np.zeros(21))
                    self.u_init.append(np.zeros(12))

                    if self.ListAction[i+2].__class__.__name__ == "ActionModelQuadrupedTime" :
                        self.x_init.append(np.zeros(21))
                        self.u_init.append(np.zeros(1))
                        self.ListAction[i+2].updateModel(np.reshape(self.l_fsteps, (3, 4), order='F') , xref[:, i+gap]  , self.gait[j, 1:] - self.gait[j-1, 1:])
                        # self.ListAction[i+1].updateModel(np.reshape(self.l_fsteps, (3, 4), order='F') , xref[:, i+gap]  , self.gait[j, 1:])
                        k_cum +=  1
                        gap -= 1

                    
                else : 
                    self.ListAction[i].updateModel(np.reshape(self.l_fsteps, (3, 4), order='F') , xref[:, i+gap]  , self.gait[j, 1:])                    
                    self.x_init.append(np.zeros(21))
                    self.u_init.append(np.zeros(12))

            k_cum += np.int(self.gait[j, 0])
            j += 1 

      
        # if k > self.start_stop_optim : 
            # Update the lastPositionweight
            # self.updatePositionWeights()
        
        
        # # Update model of the terminal model
        self.terminalModel.updateModel(np.reshape(self.fsteps[j-1, 1:], (3, 4), order='F') , xref[:,-1] , self.gait[j-1, 1:])
        self.x_init.append(np.zeros(21))
     
        # Shooting problem
        self.problem = crocoddyl.ShootingProblem(np.zeros(21),  self.ListAction, self.terminalModel)        

        self.problem.x0 = np.concatenate([xref[:,0] , p0 , [self.dt_initial]   ])

        # for elt in self.ListAction : 
        #     print(elt.__class__.__name__)
        #     data = elt.createData()
        #     x = np.random.rand(21)
        #     x[20] = np.random.rand(1)[0]
        #     u = np.random.rand(elt.nu)
        #     elt.calc(data,x,u)

        # DDP Solver
        self.ddp = crocoddyl.SolverDDP(self.problem)    

        return 0       
        
    def get_latest_result(self):
        """Return the desired contact forces that have been computed by the last iteration of the MPC
        Args:
        """
        if self.ListAction[0].__class__.__name__ == "ActionModelQuadrupedStepTime" :
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
        model.dt_weight_bound = 0.
        model.dt_min = self.dt_min
        model.dt_max = self.dt_max

        model.relative_forces = self.relative_forces

        if optim_period : 
            model.heuristicWeights =  np.zeros(8)
            model.stateWeights = np.concatenate([np.array([0.0, 0.0, ]), self.stateWeights[2:]])
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

    def create_List_model(self):
        """Create the List model using ActionQuadrupedModel()  
         The same model cannot be used [model]*(T_mpc/dt) because the dynamic changes for each nodes
        """

        j = 0
        k_cum = 0
        self.ListAction = []
        # Iterate over all phases of the gait
        # The first column of xref correspond to the current state 
        while (self.gait[j, 0] != 0):
            for i in range(k_cum, k_cum+np.int(self.gait[j, 0])):

                model = quadruped_walkgen.ActionModelQuadrupedAugmentedTime()

                # Update intern parameters
                self.update_model_augmented(model , j > 1)

                # Add model to the list of model
                self.ListAction.append(model)

            
            if np.sum(self.gait[j+1, 1:]) == 4 : # No optimisation on the first line     
            
                model = quadruped_walkgen.ActionModelQuadrupedStepTime()
                # Update intern parameters
                self.update_model_step_feet(model , j > 1)

                # Add model to the list of model
                self.ListAction.append(model)
                    
            if np.sum(self.gait[j+1, 1:]) == 2 and j >= 1 :    
                modelTime = quadruped_walkgen.ActionModelQuadrupedTime()
                # Update intern parameters
                self.update_model_step_time(modelTime , j > 1)

                # Add model to the list of model
                self.ListAction.append(modelTime)        

            
            k_cum += np.int(self.gait[j, 0])
            j += 1
        

        # Model parameters of terminal node  
        self.terminalModel = quadruped_walkgen.ActionModelQuadrupedAugmentedTime()
        self.update_model_augmented(self.terminalModel , True)
        # Weights vectors of terminal node
        self.terminalModel.forceWeights = np.zeros(12)
        self.terminalModel.frictionWeights = 0.
        self.terminalModel.heuristicWeights = np.full(8,0.0)
        self.terminalModel.lastPositionWeights =  np.full(8,0.0)

        # Shooting problem
        self.problem = crocoddyl.ShootingProblem(np.zeros(21),  self.ListAction, self.terminalModel)

        # DDP Solver
        self.ddp = crocoddyl.SolverDDP(self.problem)

    
        return 0

    def create_walking_trot(self):
        """Create the matrices used to handle the gait and initialize them to perform a walking trot

        self.gait and self.fsteps matrices contains information about the walking trot
        """

        # Number of timesteps in a half period of gait
        N = np.int(0.5 * self.T_mpc/self.dt)

        # Starting status of the gait
        # 4-stance phase, 2-stance phase, 4-stance phase, 2-stance phase
        self.gait = np.zeros((self.fsteps.shape[0], 5))
        for i in range(self.n_periods):
            self.gait[(4*i):(4*(i+1)), 0] = np.array([1, N-1, 1, N-1])
            self.fsteps[(4*i):(4*(i+1)), 0] = self.gait[(4*i):(4*(i+1)), 0]

            # Set stance and swing phases
            # Coefficient (i, j) is equal to 0.0 if the j-th feet is in swing phase during the i-th phase
            # Coefficient (i, j) is equal to 1.0 if the j-th feet is in stance phase during the i-th phase
            self.gait[4*i+0, 1:] = np.ones((4,))
            self.gait[4*i+1, [1, 4]] = np.ones((2,))
            self.gait[4*i+2, 1:] = np.ones((4,))
            self.gait[4*i+3, [2, 3]] = np.ones((2,))

        return 0

    def roll(self):
        """Move one step further in the gait cycle

        Decrease by 1 the number of remaining step for the current phase of the gait and increase
        by 1 the number of remaining step for the last phase of the gait (periodic motion)

        Add and remove corresponding model in ListAction
        """
        
        # Index of the first empty line
        index = next((idx for idx, val in np.ndenumerate(self.gait[:, 0]) if val==0.0), 0.0)[0]
     
        if np.sum(self.gait[0,1:]) == 4 : 
            self.gait[0,:] = np.zeros(5)
            index -= 1 
            self.gait = np.roll(self.gait, -1, axis=0)
        else : 
            if self.gait[0,0] == 1 : 
                self.gait[0,:] = np.zeros(5)
                index -= 1 
                self.gait = np.roll(self.gait, -1, axis=0)
            else : 
                self.gait[0,0] -= 1
        


        if np.sum(self.gait[index - 1 ,1:]) == 4 :
            self.gait[index , 0] = 1 
            self.gait[index , 1:] = np.ones(4) - self.gait[index-2 , 1:]

        else :
            if self.gait[index - 1 , 0 ] >= self.gait[index - 3 , 0 ] :
                self.gait[index , :] = np.ones(5)
            else : 
                self.gait[index - 1 , 0] += 1 


                

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
        # self.Xs[:,0 ] = np.array(self.ddp.xs[0])
        k = 1 
        gap = 1
        for elt in self.ListAction :
            if elt.__class__.__name__ != "ActionModelQuadrupedStepTime" and  elt.__class__.__name__ != "ActionModelQuadrupedTime": 
                self.Xs[:,k - gap ] = np.array(self.ddp.xs[k])
            else : 
                gap += 1
            k = k + 1  
      
        ########################################
        # Compute fsteps using the state vector
        ########################################

        j = 0
        k_cum = 0

        self.fsteps[0,0] = self.gait[0,0]

        self.ListPeriod = []

        # Iterate over all phases of the gait
        while (self.gait[j, 0] != 0):
            for i in range(k_cum, k_cum+np.int(self.gait[j, 0])):
                # print(self.Xs[12:20 , i ])
                self.fsteps[j ,1: ] = np.repeat(self.gait[j,1:] , 3)*np.concatenate([self.Xs[12:14 , k_cum ],[0.],self.Xs[14:16 , k_cum ],[0.],
                                                                                    self.Xs[16:18 , k_cum ],[0.],self.Xs[18:20 , k_cum ],[0.]])    
            # print(self.gait[j,:])
            # if np.sum(self.gait[j,1:]) == 4 and j != 0: 
            #     print(self.Xs[20 , k_cum ])
            #     T_new = self.Xs[20 , k_cum ]*self.gait[j,0]
            #     self.gait_new[j,1:] = self.gait[j,1:]                
            #     self.ListPeriod.append(T_new/self.dt)
            #     self.fsteps[j,0] = np.round(T_new/self.dt)      
            #     self.gait_new[j,0] =  np.round(T_new /self.dt)
            print(self.Xs[20 , k_cum ])   
            
            if np.sum(self.gait[j,1:]) == 4 : 
                self.gait_new[j,:] = np.ones(5)

            k_cum += np.int(self.gait[j, 0])
            j += 1       
        
        # print("gait_new")
        # print(self.gait_new)
        
        self.fsteps[j:,0] = 0.

        # Adjust the gait matrix depending on the total number of nodes        

        self.gait_old = self.gait 

        # Index of the first empty line
        # index = next((idx for idx, val in np.ndenumerate(self.gait[:, 0]) if val==0.0), 0.0)[0]

        # N_nodes_new = np.sum(self.gait_new[:,0])
        # self.gait = self.gait_new
        # if N_nodes_new > self.Max_nodes : 
        #     while(N_nodes_new != self.Max_nodes) : 
        #         if np.sum(self.gait[index - 1 ,1:]) != 4 : 
        #             if self.gait[index - 1 ,0] == 1 : 
        #                 self.gait[index - 1 , :] = np.zeros(5)
        #                 index -= 1 
        #                 N_nodes_new -= 1 
        #             else : 
        #                 self.gait[index - 1 , 0] -= 1
        #                 N_nodes_new = N_nodes_new - 1 
        #         else : 
        #             self.gait[index - 1 , :] = np.zeros(5)
        #             index -= 1  
        #             N_nodes_new -= 1
        
        # if N_nodes_new < self.Max_nodes : 
        #     while(N_nodes_new != self.Max_nodes) : 
        #         if np.sum(self.gait[index - 1 ,1:]) != 4 : 
        #                 if self.gait[index - 1 , 0] < self.gait[index-3 , 0] : 
        #                     self.gait[index - 1 , 0] += 1 
        #                     N_nodes_new += 1 
        #                 else : 
        #                     self.gait[index,:] = np.ones(5)
        #                     N_nodes_new += 1 
        #                     index += 1 
        #         else : 
        #             self.gait[index  , 0] = 1
        #             self.gait[index  , 1:] = np.ones(4) - self.gait[index - 2 , 1:]
        #             index += 1 
        #             N_nodes_new += 1

        # print(self.gait)
        # quit()

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
            # pos_tmp = np.reshape(np.array(self.oMl * (np.array([self.fsteps[index, (1+i*3):(4+i*3)]]).transpose())) , (3,1) )
            # self.o_fsteps[:2, i] = pos_tmp[0:2, 0]

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




        

    

