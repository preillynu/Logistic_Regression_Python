import numpy as np
import time
from sklearn.linear_model import LogisticRegression as LR

#open file
file_data = open('../data/data1M10.txt', 'r')

#grab first line, which has information on the dataset
file_params = file_data.readline().strip("\n").split(" ")

#set dataset parameters variables
numPoints = int(file_params[0])
numDims = int(file_params[1])
numLabels = int(file_params[2])

#make numpy matricies for dataset and labels
data = np.zeros((numPoints, numDims))
labels = np.zeros(numPoints)
test_point = np.zeros(numDims)

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

start_time = time.time()

logistic = LR(n_jobs = 8)
logistic.fit(data, labels)

end_time = time.time() - start_time

print "Time take for logistic regression: ", end_time, " seconds"
print "Number of Iterations: ", logistic.n_iter_
logistic.predict([test_point])
outfile = open('out8/out100k.txt', 'a')
outfile.write(str(end_time))
outfile.write('\n')
outfile.close()


