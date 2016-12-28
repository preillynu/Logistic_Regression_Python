#include <cuda_runtime.h>
#include <cmath>

class lrGPU
{
public:
	lrGPU(float* data_in, float* labels_in, int points, int features, int iter, float a);

	~lrGPU();

	void Cleanup();

	int BLK(int num, int blksize) {
		return (num + blksize - 1) / blksize;	
	}

    void run();
    
    int classify(float* point_in);

	int 			npoints;
	int 			nfeatures;
    int             maxIter;
    float           alpha;
    
	float 			*data;
    float           *dataT;
	float 			*labels;
    float           *weights;


	// size
	size_t data_bytes;
    size_t weight_bytes;
    size_t label_bytes;
	
	// kernel configuration
	int blocksize;

	dim3 blkDim;
    dim3 MMBlkDim;
	dim3 grdDim;
    dim3 MMGrdDim;
    dim3 MMTGrdDim;
};
