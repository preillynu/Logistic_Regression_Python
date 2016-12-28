#import GPUAdder
import numpy as np
from lr import lrGPU
import time

#open file
file_data = open('data/data100k.txt', 'r')

#grab first line, which has information on the dataset
file_params = file_data.readline().strip("\n").split(" ")

#set dataset parameters variables
numPoints = int(file_params[0])
numDims = int(file_params[1])
numLabels = int(file_params[2])

#make numpy matricies for dataset and labels
#data = np.zeros((numPoints, numDims+1),  dtype=np.float32)
data = np.zeros((numPoints*(numDims+1)),  dtype=np.float32)
#dataT = np.zeros((numPoints, numDims+1),  dtype=np.float32)
labels = np.zeros(numPoints,  dtype=np.float32)

#loop variable to index data and labels
i = 0

#read in the data and labels from the file
for line in file_data:
    split_line = line.strip("\n").split(" ")
    labels[i] = int(split_line[numDims])
    for j in range(0, numDims):
        data[i*numDims + j] = float(split_line[j])
    data[i*numDims + numDims] = 1.0
    i = i + 1

iter = 100
alpha = 0.0001

# start timer
start = time.time()

mydata = lrGPU(data, labels, numPoints, numDims, iter, alpha);
mydata.run()

# end the timer
end = time.time()

runtime = end - start

print str(end - start) +  ' s'



