import pycuda.driver as cuda
import pycuda.autoinit  # noqa
#from pycuda.compiler import SourceModule
from pycuda.compiler import compile

test_kernel = compile(
"""
    const int BLOCKSIZE = 32;
    
    float sigmoid(float in){
    return 1.0 / (1 + exp(-1 * in));
    }
    
    //Tiled version of matrix multiply
    __global__ void MatrixMultiplyKernel(float *devA, float *devB, float *devC, int rows, int cols, int k, float alpha, float beta)
    {
    //Get the thread's x and y locations for its run
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    
    //Allocate shared memory to hold parts of A and B
    __shared__ float tileA[BLOCKSIZE][BLOCKSIZE];
    __shared__ float tileB[BLOCKSIZE][BLOCKSIZE];
    
    //Use sum to get the result for a specific element
    float sum = 0.0;
    
    //Use iter to see if the loop should be run again
    int iter = 0;
    
    do{
    //Check if the x thread falls within bounds of the matrices
    if ((idy < rows) && (threadIdx.x + BLOCKSIZE*iter < k)){
    tileA[threadIdx.y][threadIdx.x] = devA[threadIdx.x + idy*k + BLOCKSIZE*iter];
    }
    else {
    tileA[threadIdx.y][threadIdx.x] = 0.0;
    }
    
    //Check if the y thread falls within bounds of the matrices
    if ((threadIdx.y + BLOCKSIZE*iter < k) && (idx < cols)){
    tileB[threadIdx.y][threadIdx.x] = devB[idx + (threadIdx.y + BLOCKSIZE*iter)*cols];
    }
    else {
    tileB[threadIdx.y][threadIdx.x] = 0.0;
    }
    
    //Sync to ensure that all of the data has been grabbed for the tiles in this warp
    __syncthreads();
    
    //Sum the elements related to the element in C corresponding to idx and idy
    for (int i = 0; i < BLOCKSIZE; i++){
    sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
    }
    
    //Iterate the number done
    iter++;
    
    //Sync the threads again to ensure they have all done their work before going through the loop to get data
    __syncthreads();
    
    //Check if the tiles have covered all of C
    } while (BLOCKSIZE*iter < k);
    
    //If the thread falls within the matrix C, fill in its element, scaled by alpha and beta
    if ((idy < rows) && (idx < cols)){
    devC[idx + idy*cols] = sum * alpha + devC[idx + idy*cols] * beta;
    }
    }
    
    __global__ void distKernel(float *devA, float *devB, float *devC, int K)
    {
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((idy < K)){
    devC[idy] = (devA[idy] - devB[idy])*(devA[idy] - devB[idy]);
    }
    }
    
    //Element wise subtraction of matrix A and B, stored in matrix C
    __global__ void sub_sigKernel(float *A, float *B, float *C, int rows)
    {
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    
    //Ensure the thread is in bounds
    if (i < rows){
    C[i] = (1.0 / (1 + exp(-1 * B[i])));
    C[i] = A[i] - C[i];
    }
    }
    """)

with open("lr_kernels.cubin", "wb") as file:
    file.write(test_kernel)
