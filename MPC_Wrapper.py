# coding: utf8

import numpy as np
import MPC
import FootstepPlanner
from multiprocessing import Process, Value, Array


class MPC_Wrapper:
    """Wrapper to run FootstepPlanner + MPC on another process

    Args:
        dt (float): Time step of the MPC
        n_steps (int): Number of time steps in one gait cycle
        k_mpc (int): Number of inv dyn time step for one iteration of the MPC
        T_gait (float): Duration of one period of gait
        multiprocessing (bool): Enable/Disable running the MPC with another process
    """

    def __init__(self, dt, n_steps, k_mpc, T_gait, multiprocessing=False):

        self.f_applied = np.zeros((12,))
        self.not_first_iter = False

        # Number of TSID steps for 1 step of the MPC
        self.k_mpc = k_mpc

        self.multiprocessing = multiprocessing
        if multiprocessing:
            self.newData = Value('b', False)
            self.newResult = Value('b', False)
            self.dataIn = Array('d', [0.0] * 328)
            self.dataOut = Array('d', [0] * 12)
            self.fsteps_future = np.zeros((20, 13))
        else:
            # Create the new version of the MPC solver object
            self.mpc = MPC.MPC(dt, n_steps, T_gait)

    def solve(self, k, fstep_planner):
        """Call either the asynchronous MPC or the synchronous MPC depending on the value of multiprocessing during
        the creation of the wrapper

        Args:
            k (int): Number of inv dynamics iterations since the start of the simulation
            fstep_planner (object): FootstepPlanner object of the control loop
        """

        if self.multiprocessing:
            # TODO: Adapt asynchronous for lower number of parameters
            raise("Error: Asynchronous MPC is not up to date")
            # self.run_MPC_asynchronous(dt, n_steps, k, T_gait, joystick, fstep_planner, interface)
        else:
            self.run_MPC_synchronous(k, fstep_planner)

        return 0

    def get_latest_result(self):
        """Return the desired contact forces that have been computed by the last iteration of the MPC

        Args:

        """

        if (self.not_first_iter):
            if self.multiprocessing:
                if self.newResult.value:
                    self.newResult.value = False
                    # Retrieve desired contact forces with through the memory shared with the asynchronous
                    return self.convert_dataOut()
                else:
                    raise ValueError("Error: something went wrong with the MPC, result not available.")
            else:
                # Directly retrieve desired contact force of the synchronous MPC object
                return self.mpc.f_applied
        else:
            # Default forces for the first iteration
            self.not_first_iter = True
            return np.array([0.0, 0.0, 8.0] * 4)

    def run_MPC_synchronous(self, k, fstep_planner):
        """Run the MPC (synchronous version) to get the desired contact forces for the feet currently in stance phase

        Args:
            k (int): Number of inv dynamics iterations since the start of the simulation
            fstep_planner (object): FootstepPlanner object of the control loop
        """

        # Run the MPC to get the reference forces and the next predicted state
        # Result is stored in mpc.f_applied, mpc.q_next, mpc.v_next

        """print(dt, n_steps, k, T_gait)
        print(np.round(interface.lC.ravel(), decimals=2))
        print(np.round(interface.abg.ravel(), decimals=2))
        print(np.round(interface.lV.ravel(), decimals=2))
        print(np.round(interface.lW.ravel(), decimals=2))
        print(interface.l_feet.ravel())
        print(joystick.v_ref.ravel())
        print(fstep_planner.fsteps)"""

        if k > 1900:
            deb=1

        self.mpc.run((k/self.k_mpc), fstep_planner.xref, fstep_planner.fsteps)

        """tmp_lC = interface.lC.copy()
        tmp_lC[2, 0] += dt * interface.lV[2, 0]
        tmp_abg = interface.abg + dt * interface.lW
        tmp_abg[2, 0] = 0.0
        tmp_lfeet = interface.l_feet - dt * interface.lV
        tmp_xref = fstep_planner.xref.copy()
        tmp_xref """

        # Output of the MPC
        self.f_applied = self.mpc.f_applied

    def run_MPC_asynchronous(self, dt, n_steps, k, T_gait, joystick, fstep_planner, interface):
        """Run the MPC (asynchronous version) to get the desired contact forces for the feet currently in stance phase

        Args:
            dt (float): Time step of the MPC
            n_steps (int): Number of time steps in one gait cycle
            k (int): Number of inv dynamics iterations since the start of the simulation
            T_gait (float): duration of one period of gait
            joystick (object): interface with the gamepad
            fstep_planner (object): FootstepPlanner object of the control loop
            interface (object): Interface object of the control loop
        """

        # If this is the first iteration, creation of the parallel process
        if (k == 0):
            p = Process(target=self.create_MPC_asynchronous, args=(self.newData, self.newResult, self.dataIn, self.dataOut))
            p.start()

        # print("Setting Data")
        self.compress_dataIn(dt, n_steps, k, T_gait, joystick, fstep_planner, interface)

        """print("Sending")
        print(dt, n_steps, k, T_gait)
        print(interface.lC.ravel())
        print(interface.abg.ravel())
        print(interface.lV.ravel())
        print(interface.lW.ravel())
        print(interface.l_feet.ravel())
        print(joystick.v_ref.ravel())
        print(fstep_planner.fsteps)"""

        self.newData.value = True

        return 0

    def create_MPC_asynchronous(self, newData, newResult, dataIn, dataOut):
        """Parallel process with an infinite loop that run the asynchronous MPC

        Args:
            newData (Value): shared variable that is true if new data is available, false otherwise
            newResult (Value): shared variable that is true if a new result is available, false otherwise
            dataIn (Array): shared array that contains the data the asynchronous MPC will use as inputs
            dataOut (Array): shared array that contains the result of the asynchronous MPC
        """

        # print("Entering infinite loop")
        while True:
            # Checking if new data is available to trigger the asynchronous MPC
            if newData.value:

                # Set the shared variable to false to avoid re-trigering the asynchronous MPC
                newData.value = False
                # print("New data detected")

                # Retrieve data thanks to the decompression function and reshape it
                dt, nsteps, k, T_gait, lC, abg, lV, lW, l_feet, xref, x0, v_ref, fsteps  = self.decompress_dataIn(dataIn)
    
                #print("Receiving")
                dt = dt[0]
                nsteps = np.int(nsteps[0])
                k = k[0]
                T_gait = T_gait[0]
                lC = np.reshape(lC, (3, 1))
                abg = np.reshape(abg, (3, 1))
                lV = np.reshape(lV, (3, 1))
                lW = np.reshape(lW, (3, 1))
                l_feet = np.reshape(l_feet, (3, 4))
                xref = np.reshape(xref, (12, nsteps+1))
                x0 = np.reshape(x0, (12, 1))
                v_ref = np.reshape(v_ref, (6, 1))
                fsteps = np.reshape(fsteps, (20, 13))

                """print(dt, nsteps, k, T_gait)
                print(lC.ravel())
                print(abg.ravel())
                print(lV.ravel())
                print(lW.ravel())
                print(l_feet.ravel())
                print(v_ref.ravel())
                print(fsteps)"""

                # Create the MPC object of the parallel process during the first iteration
                if k == 0:
                    loop_mpc = MPC.MPC(dt, nsteps)

                # Run the asynchronous MPC with the data that as been retrieved
                loop_mpc.run((k/self.k_mpc), T_gait, lC, abg, lV, lW,
                             l_feet, xref, x0, v_ref, fsteps)

                # Store the result (desired forces) in the shared memory
                self.dataOut[:] = loop_mpc.f_applied.tolist()

                # Set shared variable to true to signal that a new result is available
                newResult.value = True

        return 0

    def compress_dataIn(self, dt, n_steps, k, T_gait, joystick, fstep_planner, interface):
        """Compress data in a single C-type array that belongs to the shared memory to send data from the main control
        loop to the asynchronous MPC

        Args:
            dt (float): Time step of the MPC
            n_steps (int): Number of time steps in one gait cycle
            k (int): Number of inv dynamics iterations since the start of the simulation
            T_gait (float): duration of one period of gait
            joystick (object): interface with the gamepad
            fstep_planner (object): FootstepPlanner object of the control loop
            interface (object): Interface object of the control loop
        """

        # print("Compressing dataIn")

        # Replace NaN values by 0.0 to be stored in C-type arrays
        fstep_planner.fsteps[np.isnan(fstep_planner.fsteps)] = 0.0

        # Compress data in the shared input array
        self.dataIn[:] = np.concatenate([[dt, n_steps, k, T_gait], np.array(interface.lC).ravel(), np.array(interface.abg).ravel(),
                         np.array(interface.lV).ravel(), np.array(interface.lW).ravel(), np.array(interface.l_feet).ravel(), fstep_planner.xref.ravel(), fstep_planner.x0.ravel(), joystick.v_ref.ravel(),
                         fstep_planner.fsteps.ravel()], axis=0)

        return 0.0

    def decompress_dataIn(self, dataIn):
        """Decompress data from a single C-type array that belongs to the shared memory to retrieve data from the main control
        loop in the asynchronous MPC

        Args:
            dataIn (Array): shared array that contains the data the asynchronous MPC will use as inputs
        """

        # print("Decompressing dataIn")

        # Sizes of the different variables that are stored in the C-type array
        sizes = [0, 1, 1, 1, 1, 3, 3, 3, 3, 12, (np.int(dataIn[1])+1) * 12, 12, 6, 13*20]
        csizes = np.cumsum(sizes)

        # Return decompressed variables in a list
        return [dataIn[csizes[i]:csizes[i+1]] for i in range(len(sizes)-1)]

    def convert_dataOut(self):
        """Return the result of the asynchronous MPC (desired contact forces) that is stored in the shared memory
        """

        return np.array(self.dataOut[:])

    def roll_asynchronous(self, fsteps):
        """Move one step further in the gait cycle. Since the output of the asynchronous MPC is retrieved by
        TSID during the next call to the MPC, it should not work with the current state of the gait but with the
        gait on step into the future. That way, when TSID retrieves the result, it is consistent with the current
        state of the gait.

        Decrease by 1 the number of remaining step for the current phase of the gait and increase
        by 1 the number of remaining step for the last phase of the gait (periodic motion).
        Simplification: instead of creating a new phase if required (see roll function of FootstepPlanner) we always
        increase the last one by 1 step. That way we don't need to call other functions to predict the position of
        footstep when a new phase is created.
        """

        self.fsteps_future = fsteps.copy()

        # Index of the first empty line
        index = next((idx for idx, val in np.ndenumerate(self.fsteps_future[:, 0]) if val==0.0), 0.0)[0]

        # Create a new phase if needed or increase the last one by 1 step
        self.fsteps_future[index-1, 0] += 1.0

        # Decrease the current phase by 1 step and delete it if it has ended
        if self.fsteps_future[0, 0] > 1.0:
            self.fsteps_future[0, 0] -= 1.0
        else:
            self.fsteps_future = np.roll(self.fsteps_future, -1, axis=0)
            self.fsteps_future[-1, :] = np.zeros((13, ))

        return 0
