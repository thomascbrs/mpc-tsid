import cython
import numpy as np
cimport numpy as np

ctypedef np.float64_t DTYPE_f
ctypedef np.int64_t DTYPE_i

cdef class ContactSequencer:
    cdef public float dt, t_stance, T_gait
    cdef public np.float64_t[:, :] phases
    cdef public np.int64_t[:, :] S

    @cython.locals(t_seq=np.float64_t[:, :] , phases_seq=np.float64_t[:, :])
    cpdef int createSequence(self, float t=*)

    cpdef int updateSequence(self)