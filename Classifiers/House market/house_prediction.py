# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:\\Users\\Louis\\Documents\\Computing\\AI\\Perso\\Deep_Learning_A_Z\\Classifiers\\House market\\house_train.csv')

# The y value to predict (house price) is the last column
y = dataset.iloc[:,0].values
X = dataset.iloc[:,1:16].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 15, kernel_initializer = 'glorot_normal', activation = 'relu', input_dim = 15))

# Adding the second hidden layer
# classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'glorot_normal', activation = 'relu'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 1, epochs = 10)

# Part 3 - Making predictions and evaluating the model'''

# Predicting the Test set results
print('REAL VALUES:')
print(y_test[0])
print(y_test[1])
print(y_test[2])
print(y_test[3])
print(y_test[4])
print(y_test[5])
print(y_test[6])
print(y_test[7])
print(y_test[8])
print(y_test[9])
print("[...]")
y_pred = classifier.predict(X_test)
print('PRED VALUES:')
print(y_pred[0])
print(y_pred[1])
print(y_pred[2])
print(y_pred[3])
print(y_pred[4])
print(y_pred[5])
print(y_pred[6])
print(y_pred[7])
print(y_pred[8])
print(y_pred[9])
print("[...]")