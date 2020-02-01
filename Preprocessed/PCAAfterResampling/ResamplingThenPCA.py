#This is the code apply PCA after the sampling

import numpy as np
import csv
import pandas as pd 

from sklearn.preprocessing import StandardScaler # for standardizing the Data
from sklearn.decomposition import PCA # for PCA calculation
import matplotlib.pyplot as plt # for plotting

TestingDataFrame = pd.read_csv('train_no_pca.csv', header=None) 

TestingValues = TestingDataFrame.values

sc = StandardScaler() # creating a StandardScaler object
X_std = sc.fit_transform(TestingValues) # standardizing the data

pca = PCA(n_components = 0.95)
X_pca = pca.fit_transform(X_std)

principalDf = pd.DataFrame(data = X_pca)
print(type(principalDf))
principalDf.to_csv('pcaed_trainingData.csv', header=False)
print('success!')

