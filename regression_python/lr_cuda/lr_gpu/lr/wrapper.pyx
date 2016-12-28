import numpy as np
cimport numpy as np

assert sizeof(int)   == sizeof(np.int32_t)
assert sizeof(float) == sizeof(np.float32_t)

cdef extern from "src/lr.hh":
    cdef cppclass C_lrGPU "lrGPU":
        C_lrGPU (np.float32_t*, np.float32_t*, int, int, int, float)
        void run()
        int classify(np.float32_t*)

cdef class lrGPU:
    cdef C_lrGPU* g

    cdef int data_points
    cdef int features
    cdef int iter
    cdef float alpha

    def __cinit__(self, np.ndarray[ndim=1, dtype=np.float32_t, mode="c"] data, np.ndarray[ndim=1, dtype=np.float32_t, mode="c"] labels, int points, int features, int iters, float a):

        self.data_points = points
        self.features = features
        self.iter = iters
        self.alpha = a

        # create class
        self.g = new C_lrGPU(&data[0], &labels[0], points, features, iters, a)

    def classify(self, np.ndarray[ndim=1, dtype=np.float32_t, mode="c"] point):
        return self.g.classify(&point[0])

    def run(self):
        return self.g.run()
