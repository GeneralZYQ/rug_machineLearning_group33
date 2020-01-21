
# coding: utf-8

# ### Load traning data and do one-hot encoding on labels

# In[1]:


import csv

# every 30 lines is a different speaker. Each recording is decomposed into 9 arrays of 4 points (9 lines from the original plot)
with open('training_after_PCA_after_taking_4.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))

# generate labels for the training 
integer_encoded = []
for i in range(9):
    integer_encoded += [i] * 30

onehot_encoded = []
for elem in integer_encoded:
    letter = [0 for i in range(9)]
    letter[elem] = 1
    onehot_encoded.append(letter)

y_train=onehot_encoded
x_train=data


# ### Load testing data and do one-hot encoding on labels

# In[2]:


import csv

# every 30 lines is a different speaker. Each recording is decomposed into 9 arrays of 4 points (9 lines from the original plot)
with open('testing_after_PCA_after_taking_4.csv', newline='') as csvfile:
    data_test = list(csv.reader(csvfile))


# generate labels for the training 
integer_encoded = [0]*31 + [1]*35 + [2]*88 + [3]*44 + [4]*29 + [5]*24 + [6]*40 + [7]*50 + [8]*29

onehot_encoded = []
for elem in integer_encoded:
    letter = [0 for i in range(9)]
    letter[elem] = 1
    onehot_encoded.append(letter)

y_test=onehot_encoded
x_test=data_test
print(len(y_test))
print(len(x_test))


# ### Convert data from str to list of ints

# In[3]:


import string

def Convert_to_list_of_ints(data): 
    t=[]
    c=[]
    for elem in data:
        for i in range(9):
            temp = elem[i].strip(string.punctuation)
            temp = temp.replace(" ", "")
            temp = temp.replace("'","")
            temp = list(map(float, temp.split(",")))
            c.append(temp) #concatenate the sub arrays for each of the 9 lines to make one big array
        t.append(c)
        c=[]
    return t

x_train=Convert_to_list_of_ints(x_train)
x_test=Convert_to_list_of_ints(x_test)

print(len(x_train))
print(len(x_test))


# In[4]:


print(x_train[0])


# In[12]:


import numpy as np
import random

# shuffle training data
c = list(zip(x_train, y_train))
random.shuffle(c)
x_train, y_train = zip(*c)

## Turn the lists into np.arrays
x_train, y_train = np.array(x_train), np.array(y_train)

## we also have to convert our data into three-dimensional format
x_train = np.reshape(x_train, (x_train.shape[0], 4,9))


# In[13]:


# shuffle testing data
b = list(zip(x_test, y_test))
random.shuffle(b)
x_test, y_test = zip(*b)

## Turn the lists into np.arrays
x_test, y_test = np.array(x_test), np.array(y_test)

## we also have to convert our data into three-dimensional format
x_test = np.reshape(x_test, (x_test.shape[0], 4,9))


# ### The model

# In[22]:


from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding

model = Sequential()
model.add(LSTM(units=50,activation='relu', recurrent_activation='sigmoid', return_sequences=True,input_shape=(x_train.shape[1], 9),dropout=0.2,recurrent_dropout=0.2))
model.add(LSTM(units=50,activation='relu', recurrent_activation='sigmoid', return_sequences=True, dropout=0.2,recurrent_dropout=0.2))
model.add(LSTM(units=50, activation='relu', recurrent_activation='sigmoid',return_sequences=False, dropout=0.2,recurrent_dropout=0.2))
model.add(Dense(units=9,activation='softmax'))
model.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics = ['accuracy'])
history = model.fit(x_train, y_train, batch_size=5, epochs=30)

# saving the model
model.save('keras_ML1.h5')


# In[23]:


from keras.models import load_model

# testing the model
model = load_model('keras_ML1.h5')
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

