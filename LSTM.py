# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 18:09:36 2017

@author: Igor Molchanov
"""
# load and plot dataset
import numpy as np
import scipy as scp
import pandas as pd
import math
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# load dataset
dataset = pd.read_csv('sensor-data.csv', index_col = 0)
dataset.fillna(0, inplace=True)

#We use number of computers to reduce RMSE
df = dataset[['Nr-People','Nr-Computers','Motion', 'Brightness', 'Noise','Relative-humidity']]
values = df.values

# Ensure all data is float
values = values.astype('float32')

# Normalize features
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# X is input, y is output
x = scaled[:, 1:]   # Everything Except the first column, for 2D array
y = scaled[:, 0]    # First column only, for 2D array

train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=0.33, random_state=42)

# Reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

## Design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]),implementation=2))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# Fit network
history = model.fit(train_X, train_y, epochs=20, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
#epochs = number of going through the training set, batch_size = number of rows we look at one time
# Plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction (yhat = predictions)
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# invert scaling for forecast (inv_yhat = predictions in actual values)
inv_yhat = np.concatenate((yhat, test_X), axis=1)   # Add the inverse of the predictions to test_X to get back to the correct shape
inv_yhat = scaler.inverse_transform(inv_yhat)       # Reverse the inverse to gain normal values again
inv_yhat = inv_yhat[:,0]                            # Only interested in the first column for calculating the RMSE


# invert scaling for actual (Actual number of people in the test set)
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

validationSet = pd.read_csv('validate.csv', header=0, index_col = 0)
validationSet.fillna(0, inplace=True)
dv = validationSet[['Nr-People','Nr-Computers','Motion', 'Brightness', 'Noise','Relative-humidity']]
validationValues = dv.values
#
# Normalize features
scalerValid = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaledValid = scalerValid.fit_transform(validationValues)
#
for index, row in dv.iterrows():
    array = np.array(row)
    array = array.reshape((1, 1, array.shape[0]))
    v_x = array[:,:,1:]
    v_y = array[:,:,0]
    prediction = model.predict(v_x)
    error = sqrt(mean_squared_error(v_y, prediction))
    print('Predicted: %.3f' % prediction)
    print('Real:'+ ' ' + repr(v_y[0][0]))
    print('Validate RMSE: %.3f' % error)
    print('\n')
