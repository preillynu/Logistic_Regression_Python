#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

#include <lr.hh>
#include <kernel.cu>

using namespace std;
//lrGPU
lrGPU::lrGPU(float* data_in, float* labels_in, int points, int features, int iter, float a){

    npoints = points;
    nfeatures = features;
    maxIter = iter;
    alpha = a;

    data_bytes = npoints*(nfeatures+1)*sizeof(float);
    weight_bytes = (nfeatures+1)*sizeof(float);
    label_bytes = npoints*sizeof(float);

    if(data != NULL) cudaFree(data);
    cudaMallocManaged((void **)&data, data_bytes);
    cudaMemcpy(data, data_in, data_bytes, cudaMemcpyHostToDevice);

    float *dT = new float[npoints*(nfeatures+1)];
    for (int i = 0; i < npoints; i++){
        for (int j = 0; j < nfeatures+1; j++){
            dT[j*npoints + i] = data_in[j+i*(nfeatures+1)];
        }
    }

    if(dataT != NULL) cudaFree(dataT);
    cudaMallocManaged((void **)&dataT, data_bytes);
    cudaMemcpy(dataT, dT, data_bytes, cudaMemcpyHostToDevice);

    if(labels != NULL) cudaFree(labels);
    cudaMallocManaged((void **)&labels, label_bytes);
    cudaMemcpy(labels, labels_in, label_bytes, cudaMemcpyHostToDevice);

    float* randWeights = new float[nfeatures+1];

    for (int i = 0; i < nfeatures+1; i++){
        randWeights[i] = ((double)rand() / (RAND_MAX));
    }

    if(weights != NULL) cudaFree(weights);
    cudaMallocManaged((void **)&weights, weight_bytes);
    cudaMemcpy(weights, randWeights, weight_bytes, cudaMemcpyHostToDevice);

    blocksize = 32;
    blkDim = dim3(1, blocksize*blocksize, 1);
    grdDim = dim3(1, BLK(npoints, blocksize*blocksize), 1);
    MMBlkDim = dim3(blocksize, blocksize, 1);
    MMGrdDim = dim3(1, BLK(npoints, blocksize), 1);
    MMTGrdDim = dim3(1, BLK(nfeatures, blocksize), 1);

}

lrGPU::~lrGPU()
{
    Cleanup();
}

void lrGPU::Cleanup()
{
    if(data != NULL) cudaFree(data);
    if(labels != NULL) cudaFree(labels);
    if(weights != NULL) cudaFree(weights);
}

void lrGPU::run()
{
    float *error, *prob;
    cudaMallocManaged((void **)&error, label_bytes);
    cudaMallocManaged((void **)&prob, label_bytes);
    float *oldWeights; cudaMallocManaged((void **)&oldWeights, weight_bytes);
    float *change; cudaMallocManaged((void **)&change, weight_bytes);
    float tol = 0.0001;
    float *check = new float[nfeatures+1];
    float sum = 0.0;
    int grid = BLK(nfeatures, blocksize);

    for (int i = 0; i < maxIter; i++){
		
		cudaMemcpy(oldWeights, weights, weight_bytes, cudaMemcpyDeviceToDevice);
		sum = 0.0;
	
		MatrixMultiplyKernel<<<MMGrdDim, MMBlkDim>>>(data, weights, prob, npoints, 1, nfeatures + 1, 1.0, 0.0);
		sub_sigKernel<<<grdDim, blkDim>>>(labels, prob, error, npoints);
		MatrixMultiplyKernel <<<MMTGrdDim, MMBlkDim>>>(dataT, error, weights, nfeatures + 1, 1, npoints, alpha, 1.0);
		
		distKernel<<<dim3(1,grid,1), dim3(1, blocksize, 1)>>>(weights, oldWeights, change, nfeatures+1);
		cudaMemcpy(check, change, weight_bytes, cudaMemcpyDeviceToHost);
		
		for(int j = 0; j < nfeatures + 1; j++){
			sum += check[j];
		}	
		sum = sqrt(sum);
		alpha = alpha - alpha/(maxIter - i);
		
		if(sum < (tol*(nfeatures+1))){
			cout << i+1 << " iterations" << endl;
			break;
		}
        }
    cout << sum << " final dist" << endl;

    cudaFree(error); cudaFree(prob); cudaFree(oldWeights); cudaFree(change);
}

int lrGPU::classify(float* point_in)
{
    //Use classify to classify the point
    float classify = 0.0;
    float *updatedWeights;

    //Copy the weights back to the host
    cudaMallocManaged((void **)&updatedWeights, (nfeatures+1)*sizeof(float));
    cudaMemcpy(updatedWeights, weights, (nfeatures + 1)*sizeof(float), cudaMemcpyDeviceToHost);

    //Classify the point by summing the products of corresponding weights and data dimensions
    for (int k = 0; k < nfeatures + 1; k++){
        classify += updatedWeights[k] * point_in[k];
    }

    //Classify the point
    classify = sigmoid(classify);
    if (classify > 0.5){
        return 1;
    }else{
        return 0;
    }
}
