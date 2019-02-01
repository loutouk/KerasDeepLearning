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
dataset = pd.read_csv('C:\\Users\\Louis\\Documents\\Computing\\AI\\Perso\\Deep_Learning_A_Z\\Kaggle_bank_compet\\iris_train.csv')

y = dataset.iloc[:,0:3].values

X = dataset.iloc[:,3:7].values
# Taking care of missing data
# changing nan value to the mean of the column
X = pd.DataFrame(X).fillna(X.mean())

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
classifier.add(Dense(units = 4, kernel_initializer = 'glorot_normal', activation = 'relu', input_dim = 4))

# Adding the output layer
classifier.add(Dense(units = 3, kernel_initializer = 'glorot_normal', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 8, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
print("Real values")
print(y_test)
y_pred = classifier.predict(X_test)
print("predictions")
print(y_pred)
