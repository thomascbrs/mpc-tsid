import cython
cimport ContactSequencer as CS
import numpy as np
cimport numpy as np

ctypedef np.float64_t DTYPE_f

cdef class FootstepPlanner:
    cdef public float k_feedback, dt, g, L, 
    cdef public np.float64_t[:, :] shoulders, footsteps, footsteps_world, footsteps_prediction, footsteps_tsid, t_remaining_tsid


    @cython.locals(index=int)
    cpdef object update_footsteps_tsid(self, CS.ContactSequencer sequencer, np.ndarray[dtype=DTYPE_f, ndim=2] vel_ref, np.ndarray[dtype=DTYPE_f, ndim=2] v_xy, float t_stance, float T, float h)
