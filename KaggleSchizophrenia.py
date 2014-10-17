import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.lda import LDA

# Configuration
# Output file name
outputFile = 'schizoSolution.csv';
# fMRI FNC mapping file
spatialPosFile = 'AdditionalInformation/rs_fMRI_FNC_mapping.csv'
# Training features file
trainFile = 'train_FNC.csv'
# Test features file
testFile = 'test_FNC.csv'
# Number of spatial filters
m = 3
# Regularization
a = 0

# Load spatial position file
spatialLocation = pd.read_csv(spatialPosFile, delimiter=",").values
# Remove IDs
spatialLocation = np.delete(spatialLocation, 0, axis=1)


# Load train data
print 'Loading train data ...'
trainData = pd.read_csv(trainFile, delimiter=",").values
# Remove IDs
trainData = np.delete(trainData, 0, axis=1)

# Load test data
print 'Loading test data ...'
testData = pd.read_csv(testFile, delimiter=",").values
# Save IDs
testIDs = testData[:,0].astype(np.int)
# Remove IDs
testData = np.delete(testData, 0, axis=1)

# Channel positions
positions = np.unique(spatialLocation)

# Build spatial covariances
print 'Building spatial covariances ...'

Nfeatures = trainData.shape[1]
Nsubjects = trainData.shape[0]
Nchannels = positions.shape[0]

# Building training covariances
print 'Train spatial covariaces ...'
K = np.zeros((Nsubjects, Nchannels, Nchannels))
for i in range(trainData.shape[1]):
    f = trainData[:, i] 
    ind1 = np.argwhere( spatialLocation[i, 0] == positions )[0][0]
    ind2 = np.argwhere( spatialLocation[i, 1] == positions )[0][0]
    K[:, ind1, ind2] = f
    K[:, ind2, ind1] = f

for i in range(Nchannels):
    K[:, i, i] = 1
    
# Building training covariances
print 'Test spatial covariaces ...'
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
print 'Training CSP. Computing spatial filters ...'
labels = np.genfromtxt('train_labels.csv', delimiter = ',')
# Remove fields
labels = np.delete(labels, 0, axis=0)
# Remove IDs
labels = np.delete(labels, 0, axis=1)
labels = labels[:,0]

# CSP by generalized eigenvalue problem
mean1 = np.mean(K[labels == 1,:,:], axis = 0)
mean2 = np.mean(K[labels == 0,:,:], axis = 0)
D, W = linalg.eig(mean1, mean2 + a*np.eye(mean1.shape[0]))
D = D.real # Removing spurious imag part
W = W.real

# Filters with higher and lower eigenvalue are selected
ind = np.argsort(D)
sel = np.hstack( (np.arange(0, m), np.arange(W.shape[0] - m, W.shape[0])) )
W = W[:, ind[sel]]

print 'Spatial filtering ...'

# Spatial filtering train
print 'Filtering train data ...'
trainFeats = np.empty([K.shape[0], m*2])
for i in range(0,K.shape[0]):
    aux = np.dot( np.dot(W.T, K[i,:,:]), W )
    trainFeats[i,:] = ( np.diag(aux) ) / np.trace(aux)

# Spatial filtering test
print 'Filtering test data ...'
testFeats = np.empty([Ktest.shape[0], m*2])
for i in range(0,Ktest.shape[0]):
    aux = np.dot( np.dot(W.T, Ktest[i,:,:]), W )
    testFeats[i,:] = ( np.diag(aux) ) / np.trace(aux)


# Classification
print 'Classification ...'
clf = LDA()
clf.fit(trainFeats, labels)
predictedProb = clf.predict_proba(testFeats)
predictedProb = predictedProb[:,1]

# Generate submission
submission = {'ID' : testIDs, 
              'probability' : predictedProb }
submission = pd.DataFrame(submission)
submission.to_csv(outputFile, index = 0, float_format='%11.6f')



