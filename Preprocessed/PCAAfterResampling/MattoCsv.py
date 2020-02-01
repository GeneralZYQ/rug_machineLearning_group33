#This is the file used to transfer the mat to .csv
import scipy.io
import numpy as np
data = scipy.io.loadmat("vanilla_interpolation_test5.mat")

for i in data:
    if '__' not in i and 'readme' not in i:
          np.savetxt(("vanilla_interpolation_test5.csv"),data[i],delimiter=',')