#This file is used to firstly pick the points from lines then apply PCA on them

import numpy as np
import csv
import pandas as pd 

from sklearn.preprocessing import StandardScaler # for standardizing the Data
from sklearn.decomposition import PCA # for PCA calculation
import matplotlib.pyplot as plt # for plotting


bins = 6;

#Produce the input features
traningFeaturesReading = open('ae.train', 'r+')
allmanFeatures = []
oneManFeatures = []
while True:

    line = traningFeaturesReading.readline()

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

allSimplifiedMan = []
frames = []
for oneMan in allmanFeatures:
    timeLength = len(oneMan)
    simplifiedMan = []
    step = timeLength / (bins - 1); 
    for i in range(0, 12):#because there are 12 lines
        simplifiedMan.append([])
        for b in range(0,bins):
            bias = round(b*step) if round(b*step) < timeLength else -1;
            simplifiedMan[-1].append(oneMan[bias][i])

    df = pd.DataFrame(simplifiedMan)
    df = df.transpose()
    frames.append(df)

    allSimplifiedMan.append(simplifiedMan)

TrainingFlattendFeatures = pd.concat(frames) #this is the picked data of all.


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



allTestSimplifiedMan = []
Testframes = []
for oneMan in alltestFeatures:
    timeLength = len(oneMan)
    simplifiedMan = []
    step = timeLength / (bins - 1); 
    for i in range(0, 12):#because there are 12 lines
        simplifiedMan.append([])
        for b in range(0,bins):
            bias = round(b*step) if round(b*step) < timeLength else -1;
            simplifiedMan[-1].append(oneMan[bias][i])

    df = pd.DataFrame(simplifiedMan)
    df = df.transpose()
    Testframes.append(df)

    allTestSimplifiedMan.append(simplifiedMan)

TestingFlattendFeatures = pd.concat(Testframes) #this is the picked data of all.

X = TrainingFlattendFeatures.values # getting all values as a matrix of dataframe 
X_test = TestingFlattendFeatures.values

sc = StandardScaler() # creating a StandardScaler object
X_std = sc.fit_transform(X) # standardizing the data
X_test_std = sc.fit_transform(X_test)

pca = PCA(n_components = 0.95)
X_pca = pca.fit_transform(X_std) # this will fit and reduce dimensions
X_pca_test = pca.fit_transform(X_test_std)

principalDf = pd.DataFrame(data = X_pca)
principalTestDf = pd.DataFrame(data=X_pca_test)

AllTraningDataAppliedPCA = []
for i in range(0,270):
    onePar = principalDf.iloc[0:6, :]
    parlist = onePar.values.tolist()
    AllTraningDataAppliedPCA.append(parlist)

PCADataframe = pd.DataFrame(AllTraningDataAppliedPCA)
PCADataframe.to_csv('CPA_forward_training_taking_6.csv')

AllTestingDataAppliedPCA = []
print(principalTestDf.shape[0])
TestingCount = round(principalTestDf.shape[0]/ 6) 
for i in range(0, TestingCount):
    onePar = principalTestDf.iloc[0:6, :]
    parlist = onePar.values.tolist()
    AllTestingDataAppliedPCA.append(parlist)

TestingPCADF = pd.DataFrame(AllTestingDataAppliedPCA)
TestingPCADF.to_csv('CPA_forward_testing_taking_6.csv')







