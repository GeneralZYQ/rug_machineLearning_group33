

#Import libraries
import math 

import pandas as pd # for using pandas daraframe
import numpy as np # for som math operations
from sklearn.preprocessing import StandardScaler # for standardizing the Data
from sklearn.decomposition import PCA # for PCA calculation
import matplotlib.pyplot as plt # for plotting

#Produce the input features
fotraning = open('ae.train', 'r+')
allmanFeatures = []
oneManFeatures = []
while True:

    line = fotraning.readline()

    if not line:
        break;

    components = line.split(' ')
    del components[-1]

    if len(components) == 13:
        del components[-1]
    
    if float(components[0]) != 1.0:
        oneManFeatures.append(components)
    else:
        allmanFeatures.append(oneManFeatures);
        oneManFeatures = [];


trainingLabels = []
for i in range(0,270):
    featureIndex = math.ceil(i / 270.0)
    trainingLabels.append(featureIndex)


#Produce the testing features
fotesting = open('ae.test', 'r+')
alltestFeatures = []
oneTestFeatures = []
while True:

    line = fotesting.readline()

    if not line:
        break;

    components = line.split(' ')
    del components[-1]

    if len(components) == 13:
        del components[-1]
    
    if len(components) > 0:
        if float(components[0]) != 1.0:
            oneTestFeatures.append(components)
        else:
            alltestFeatures.append(oneTestFeatures);
            oneTestFeatures = [];

# print (len(alltestFeatures))

#produce the testing labels
testingLabels = []
blockLengthes = [31, 35, 88, 44, 29, 24, 40, 50, 29];


total = 0
for j in range(0,len(blockLengthes)):
    num = blockLengthes[j];
    for k in range(total, total + num):
        testingLabels.append(j + 1);

# print (len(testingLabels))


flattenedFeatures = [] #Used for PCA
for i in range(0,270):
    fs = allmanFeatures[i]
    for j in range(0, len(fs)):
        flattenedFeatures.append(fs[j])

TrainingDataFrame = pd.DataFrame(flattenedFeatures, columns =['channel1', 'channel2', 'channel3', 'channel4', 'channel5', 'channel6', 'channel7', 'channel8', 'channel9', 'channel10', 'channel11', 'channel12'])

flattenedTestFeatures = [] #Used for PCA too.
for i in range(0, sum(blockLengthes)):
    fs = alltestFeatures[i]
    for j in range(0, len(fs)):
        flattenedTestFeatures.append(fs[j])

TestingDataFrame = pd.DataFrame(flattenedTestFeatures, columns =['channel1', 'channel2', 'channel3', 'channel4', 'channel5', 'channel6', 'channel7', 'channel8', 'channel9', 'channel10', 'channel11', 'channel12'])

X = TrainingDataFrame.values # getting all values as a matrix of dataframe 
X_test = TestingDataFrame.values
sc = StandardScaler() # creating a StandardScaler object
X_std = sc.fit_transform(X) # standardizing the data
X_test_std = sc.fit_transform(X_test)

pca = PCA(n_components = 0.95)
X_pca = pca.fit_transform(X_std) # this will fit and reduce dimensions
X_pca_test = pca.fit_transform(X_test_std)
# print(pca.n_components_) # one can print and see how many components are selected. In this case it is 4 same as above we saw in step 5
principalDf = pd.DataFrame(data = X_pca)
principalTestDf = pd.DataFrame(data=X_pca_test)
print (principalTestDf.shape)


starter = 0
processedTrainingData = []
for i in range(0,270):
    oneFeatures = allmanFeatures[i]
    datalength = len(oneFeatures)
    oneManProcessed = []
    for j in range(0, principalDf.shape[1]):
        onecoloum = principalDf.iloc[starter:starter+datalength, j]
        oneManProcessed.append(list(onecoloum.values))
    starter = starter+datalength
    processedTrainingData.append(oneManProcessed)

processedTraningDataFrame = pd.DataFrame(processedTrainingData)
print(processedTraningDataFrame.shape)
processedTraningDataFrame.to_csv('training.csv')



testingStarter = 0
processedTestingData = []
for i in range(0, sum(blockLengthes)):
    oneTestFeatures = alltestFeatures[i]
    datalength = len(oneTestFeatures)
    oneTestingProcessed = []
    for j in range(0, principalTestDf.shape[1]):
        onecoloum = principalTestDf.iloc[testingStarter:testingStarter+datalength, j]
        oneTestingProcessed.append(list(onecoloum.values))
    testingStarter = testingStarter + datalength
    processedTestingData.append(oneTestingProcessed)

processedTestngDataFrame = pd.DataFrame(processedTestingData)
print(processedTestngDataFrame.shape)
processedTestngDataFrame.to_csv('testing.csv')


        # print(principalDf.iloc[0:25, :]) 
        
#     leng = len(oneManFeatures);
#     print(leng)

# n_pcs= pca.n_components_ # get number of component ,
# most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)] #get the index of the most important feature on EACH component
# print (most_important)

#Top 99% important features: [3, 9, 4, 0, 6, 10, 1, 4, 3, 11, 7] (There is no channel3)
#Top 90% important featres: [3, 9, 4, 0, 6, 10, 1, 4] (There is no channel3, channel4, channel6, channel12)


# for i in range(0, 270):
#     print (i)
#     TrainingDataFrame = pd.DataFrame(allmanFeatures[i], columns =['channel1', 'channel2', 'channel3', 'channel4', 'channel5', 'channel6', 'channel7', 'channel8', 'channel9', 'channel10', 'channel11', 'channel12'])


    
    
