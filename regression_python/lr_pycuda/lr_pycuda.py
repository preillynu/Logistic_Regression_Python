import pycuda.driver as cuda
import pycuda.autoinit
# from pycuda.compiler import SourceModule
from pycuda.driver import module_from_file

import pycuda.gpuarray as gpuarray
import numpy as np

import time

### ------------------------------------------
### start timing the start of the end-to-end processing time
### ------------------------------------------

## load precompiled cubin file
mod = module_from_file("lr_kernels.cubin")

# link to the kernel function

lr_MM = mod.get_function('MatrixMultiplyKernel')
lr_sig = mod.get_function('sub_sigKernel')
lr_dist = mod.get_function('distKernel')


#------------------------------------------------------------------------------
# parameters
#------------------------------------------------------------------------------
maxIter = 100
alpha = 0.001
tol = 0.0001

# input data

#open file
file_data = open('../data/data1M10.txt', 'r')

#grab first line, which has information on the dataset
file_params = file_data.readline().strip("\n").split(" ")

#set dataset parameters variables
numPoints = int(file_params[0])
numDims = int(file_params[1])
numLabels = int(file_params[2])

#make numpy matricies for dataset and labels
data = np.zeros((numPoints, numDims+1)).astype('f')
dataT = np.zeros((numDims+1, numPoints)).astype('f')
labels = np.zeros(numPoints).astype('f')
test_point = np.zeros(numDims+1).astype('f')
weights = np.random.rand(numDims+1).astype('f')
oldWeights = np.zeros(numDims+1).astype('f')
distances = np.zeros(numDims+1).astype('f')

#loop variable to index data and labels
i = 0

#read in the data and labels from the file
for line in file_data:
    split_line = line.strip("\n").split(" ")
    labels[i] = int(split_line[numDims])
    for j in range(0, numDims):
        data[i][j] = float(split_line[j])
    i = i + 1

file_data.close()

start = time.time()

for i in range(0,numPoints):
    for j in range(0, numDims+1):
        dataT[j][i] = data[i][j]



###
## allocate memory on device
###

X_gpu = cuda.mem_alloc(data.nbytes)
X_t_gpu = cuda.mem_alloc(dataT.nbytes)
weights_gpu = cuda.mem_alloc(weights.nbytes)
old_weights_gpu = cuda.mem_alloc(weights.nbytes)
distances_gpu = cuda.mem_alloc(weights.nbytes)
labels_gpu = cuda.mem_alloc(labels.nbytes)
error_gpu = cuda.mem_alloc(labels.nbytes)
prob_gpu = cuda.mem_alloc(labels.nbytes)

###
## transfer data to gpu
###
cuda.memcpy_htod(X_gpu, data)
cuda.memcpy_htod(X_t_gpu, dataT)

cuda.memcpy_htod(weights_gpu, weights)

cuda.memcpy_htod(labels_gpu, labels)

###
## define kernel configuration
###
blk_size = 32
grd_size = (numPoints + blk_size -1) / blk_size
grd_size_T = (numDims + blk_size) / blk_size
grd_size_sig = (numPoints + blk_size*blk_size -1) / (blk_size*blk_size)

###---------------------------------------------------------------------------
### Run kmeans on gpu
###---------------------------------------------------------------------------

sum = 0.0

for i in range(0, maxIter):
        cuda.memcpy_dtod(old_weights_gpu, weights_gpu, weights.nbytes)
        lr_MM(X_gpu, weights_gpu, prob_gpu, \
              np.int32(numPoints), np.int32(1), np.int32(numDims + 1), np.float32(1.0), np.float32(0.0),\
              block = (blk_size, blk_size, 1), grid = (1, grd_size, 1))
        lr_sig(labels_gpu, prob_gpu, error_gpu, \
               np.int32(numPoints), \
               block = (1, blk_size * blk_size, 1), grid = (1, grd_size_sig, 1))
        lr_MM(X_t_gpu, error_gpu, weights_gpu, \
          np.int32(numDims+1), np.int32(1), np.int32(numPoints), np.float32(alpha), np.float32(1.0),\
          block = (blk_size, blk_size, 1), grid = (1, grd_size_T, 1))

        lr_dist(weights_gpu, old_weights_gpu, distances_gpu, \
                  np.int32(numDims + 1), \
                  block = (1, blk_size, 1), grid = (1, grd_size_T, 1))
        cuda.memcpy_dtoh(distances, distances_gpu)
        for j in range(0, numDims+1):
            sum = sum + distances[j]

        sum = sum ** (0.5)
        alpha = alpha - alpha/(maxIter - i)
        if (sum < tol*(numDims+1)):
            break

### ------------------------------------------
### end timing of the end-to-end processing time
### ------------------------------------------
end = time.time()
runtime = end - start

###----------------------------------------------------------------------------
## dump stat
###----------------------------------------------------------------------------


print 'runtime : ' + str(runtime)  + ' s'
