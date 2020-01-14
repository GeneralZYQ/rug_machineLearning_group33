#This file is used as the interpolation processing.
import numpy as np

# Data_reduced = np.load('training.csv',allow_pickle=True) #or something

Data_reduced2 = [] #want this to have 270 rows, each column is an array of length 4, should have 9 columns (as we reduced 12 lines to 9 lines in PCA)
for i in range(1,270): #later should do 370 for test data
    Data_reduced2.append([])
    for l in range(1,9):
        Series_PCA = Data_reduced[i,l] #select one time series/recording from input data after PCA has been performed. Starts on second line i.e. index 1. I also just want the array
        Series_PCA = Series_PCA.split(',')
        Series_length = length(Series_PCA)
        Series_reduced = [Series_PCA[1],Series_PCA[round(Series_length/3)],Series_PCA[round(2*Series_length/3)],Series_PCA[Series_length]]
        #Data_reduced2(i-1,l).append(Series_reduced)
        Data_reduced2[-1].append(Series_reduced)
#need to step through data - how is it organised? Will also need to ensure corresponding labels make sense.