#This is the code apply PCA after the sampling

import numpy as np
import csv
import pandas as pd 

from sklearn.preprocessing import StandardScaler # for standardizing the Data
from sklearn.decomposition import PCA # for PCA calculation
import matplotlib.pyplot as plt # for plotting

TrainingDataFrame = pd.read_csv('train_no_pca.csv', header=None) 
TestingDataFrame = pd.read_csv('vanilla_interpolation_test5.csv')

TrainingValues = TrainingDataFrame.values
TestingValues = TestingDataFrame.values

sc = StandardScaler() # creating a StandardScaler object
X_std = sc.fit_transform(TrainingValues) # standardizing the data
Test_std = sc.fit_transform(TestingValues)

pca = PCA(n_components = 0.95)
X_pca = pca.fit_transform(X_std)
X_pca_test = pca.fit_transform(Test_std)

principalDf = pd.DataFrame(data = X_pca)
principalTestDF = pd.DataFrame(data=X_pca_test)
print(pca.explained_variance_ratio_)
principalDf.to_csv('pcaed_trainingData.csv', header=False)
principalTestDF.to_csv('vanilla_interpolation_test5_pca.csv')
print('success!')

