#This is the code apply PCA after the sampling

import numpy as np
import csv
import pandas as pd 

from sklearn.preprocessing import StandardScaler # for standardizing the Data
from sklearn.decomposition import PCA # for PCA calculation
import matplotlib.pyplot as plt # for plotting

TrainingDataFrame = pd.read_csv('vanilla_interpolation_train5.csv', header=None) 
TestingDataFrame = pd.read_csv('vanilla_interpolation_test5.csv', header=None)

TrainingValues = TrainingDataFrame.values
TestingValues = TestingDataFrame.values

sc = StandardScaler() # creating a StandardScaler object
X_std = sc.fit_transform(TrainingValues) # standardizing the data
Test_std = sc.fit_transform(TestingValues)

pca = PCA(n_components = 0.99)
X_pca = pca.fit_transform(X_std)
X_pca_test = pca.fit_transform(Test_std)

principalDf = pd.DataFrame(data = X_pca)
principalTestDF = pd.DataFrame(data=X_pca_test)

n_pcs= pca.n_components_
most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
initial_feature_names = TrainingDataFrame.columns
# get the most important feature names
most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]
print(most_important_names)
most_important_names = list(set(most_important_names))


most_important_training_df = TrainingDataFrame.iloc[:,most_important_names]
most_important_training_df.to_csv('vanilla_most_important_training_features_099.csv', header=False)
most_important_testing_df = TestingDataFrame.iloc[:,most_important_names]
most_important_testing_df.to_csv('vanilla_most_important_testing_features_099.csv', header=False)


# print(pca.explained_variance_ratio_)
# print(pca.components_)
# principalDf.to_csv('vanilla_interpolation_train5_pca_1.csv', header=False)
# principalTestDF.to_csv('vanilla_interpolation_test5_pca_1.csv',header = False)
# print('success!')

