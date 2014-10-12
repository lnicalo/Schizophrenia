import numpy as np
import pandas as pd
from scipy import linalg

# Load spatial position file
spatialLocation = pd.read_csv("AdditionalInformation/rs_fMRI_FNC_mapping.csv", delimiter=",").values
# Remove IDs
spatialLocation = np.delete(spatialLocation, 0, axis=1)

# Load train data
trainData = pd.read_csv("./train_FNC.csv", delimiter=",").values
# Remove IDs
trainData = np.delete(trainData, 0, axis=1)

# Load test data
testData = pd.read_csv("./test_FNC.csv", delimiter=",").values
# Remove IDs
testData = np.delete(testData, 0, axis=1)


# Channel positions
positions = np.unique(spatialLocation)

# Build spatial covariances
Nfeatures = trainData.shape[1]
Nsubjects = trainData.shape[0]
Nchannels = positions.shape[0]

K = np.zeros((Nsubjects, Nchannels, Nchannels))
for i in range(trainData.shape[1]):
    f = trainData[:, i] 
    ind1 = np.argwhere( spatialLocation[i, 0] == positions )[0][0]
    ind2 = np.argwhere( spatialLocation[i, 1] == positions )[0][0]
    K[:, ind1, ind2] = f
    K[:, ind2, ind1] = f

for i in range(Nchannels):
    K[:, i, i] = 1
    

# Build spatial covariances
NtestSubjects = testData.shape[0]
Nchannels = positions.shape[0]

Ktest = np.zeros((NtestSubjects, Nchannels, Nchannels))
for i in range(testData.shape[1]):
    f = testData[:, i] 
    ind1 = np.argwhere( spatialLocation[i, 0] == positions )[0][0]
    ind2 = np.argwhere( spatialLocation[i, 1] == positions )[0][0]
    Ktest[:, ind1, ind2] = f
    Ktest[:, ind2, ind1] = f

for i in range(Nchannels):
    Ktest[:, i, i] = 1


# Load labels
labels = np.genfromtxt('train_labels.csv', delimiter = ',')
# Remove fields
labels = np.delete(labels, 0, axis=0)
# Remove IDs
labels = np.delete(labels, 0, axis=1)
labels = labels[:,0]

# CSP by generalized eigenvalue problem
mean1 = np.mean(K[labels == 1,:,:], axis = 0)
mean2 = np.mean(K[labels == 0,:,:], axis = 0)

m = 3 # Number of spatial filters
a = 0 # Regularization
D, W = linalg.eig(mean1, mean2 + a*np.eye(mean1.shape[0]))
D = D.real # Removing spurious imag part
W = W.real

# Filters with higher and less eigenvalue are selected
ind = np.argsort(D)
sel = np.hstack( (np.arange(0, m), np.arange(W.shape[0] - m, W.shape[0])) )
W = W[:, ind[sel]]

# Spatial filtering train
trainFeats = np.empty([K.shape[0], m*2])
for i in range(0,K.shape[0]):
    aux = np.dot( np.dot(W.T, K[i,:,:]), W )
    trainFeats[i,:] = ( np.diag(aux) ) # / np.trace(aux)

# Spatial filtering test
testFeats = np.empty([Ktest.shape[0], m*2])
for i in range(0,Ktest.shape[0]):
    aux = np.dot( np.dot(W.T, Ktest[i,:,:]), W )
    testFeats[i,:] = ( np.diag(aux) ) # / np.trace(aux)