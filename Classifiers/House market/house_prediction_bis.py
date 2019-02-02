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
dataset = pd.read_csv('C:\\Users\\Louis\\Documents\\Computing\\AI\\Perso\\Deep_Learning_A_Z\\Classifiers\\House market\\train.csv')

# Replace nan value by the mean of the col
dataset = dataset.fillna(dataset.mean())
#dataset = dataset.fillna(0)

# The y value to predict (house price) is the last column
y = dataset.iloc[:,80].values

# Remove the sale price as we will use the dataset frame for the X values
dataset = dataset.drop('SalePrice',1)
# Excluding the ID field as it is not relevant
dataset = dataset.drop('Id',1)
# Excluding columns that may no be necessary to facilitate the learning
dataset = dataset.drop('LotArea',1)
dataset = dataset.drop('LotFrontage',1)
dataset = dataset.drop('Street',1)
dataset = dataset.drop('LotShape',1)
dataset = dataset.drop('LandContour',1)
dataset = dataset.drop('LandSlope',1)
dataset = dataset.drop('YearRemodAdd',1)
dataset = dataset.drop('RoofStyle',1)
dataset = dataset.drop('Exterior1st',1)
dataset = dataset.drop('Exterior2nd',1)
dataset = dataset.drop('MasVnrType',1)
dataset = dataset.drop('Foundation',1)
dataset = dataset.drop('BsmtQual',1)
dataset = dataset.drop('BsmtFinType1',1)
dataset = dataset.drop('BsmtFinSF1',1)
dataset = dataset.drop('BsmtFinType2',1)
dataset = dataset.drop('BsmtFinSF2',1)
dataset = dataset.drop('BsmtUnfSF',1)
dataset = dataset.drop('TotalBsmtSF',1)
dataset = dataset.drop('LowQualFinSF',1)
dataset = dataset.drop('Fireplaces',1)
dataset = dataset.drop('FireplaceQu',1)
dataset = dataset.drop('GarageFinish',1)
dataset = dataset.drop('PavedDrive',1)
dataset = dataset.drop('MoSold',1)
dataset = dataset.drop('Alley',1)
dataset = dataset.drop('LotConfig',1)
dataset = dataset.drop('Condition1',1)
dataset = dataset.drop('Condition2',1)
dataset = dataset.drop('BldgType',1)
dataset = dataset.drop('HouseStyle',1)
dataset = dataset.drop('RoofMatl',1)
dataset = dataset.drop('BsmtCond',1)
dataset = dataset.drop('BsmtExposure',1)
dataset = dataset.drop('Electrical',1)
dataset = dataset.drop('GarageType',1)
dataset = dataset.drop('PoolQC',1)
dataset = dataset.drop('Fence',1)
dataset = dataset.drop('MiscFeature',1)
dataset = dataset.drop('ExterCond',1)
dataset = dataset.drop('BsmtFullBath',1)
dataset = dataset.drop('BsmtHalfBath',1)
dataset = dataset.drop('FullBath',1)
dataset = dataset.drop('HalfBath',1)
dataset = dataset.drop('KitchenQual',1)
dataset = dataset.drop('TotRmsAbvGrd',1)
dataset = dataset.drop('Functional',1)
dataset = dataset.drop('SaleType',1)
dataset = dataset.drop('PoolArea',1)
dataset = dataset.drop('ScreenPorch',1)
dataset = dataset.drop('3SsnPorch',1)
dataset = dataset.drop('EnclosedPorch',1)
dataset = dataset.drop('OpenPorchSF',1)

# Encoding categorical data
# Procesing categorical variables to binary 
dataset_edited = pd.get_dummies(dataset, columns=["ExterQual", "MSZoning", "Utilities", "Neighborhood", "Heating", "HeatingQC", "CentralAir","GarageCond","GarageQual","SaleCondition"])
dataset_edited = dataset_edited.apply(pd.to_numeric)

X = dataset_edited.iloc[:,:].values

#pd.DataFrame(X).to_csv("C:\\Users\\Louis\\Documents\\Computing\\AI\\Perso\\Deep_Learning_A_Z\\Classifiers\\House market\\train_saved.csv")

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#pd.DataFrame(X_train).to_csv("C:\\Users\\Louis\\Documents\\Computing\\AI\\Perso\\Deep_Learning_A_Z\\Classifiers\\House market\\train_norm_saved.csv")

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 81, kernel_initializer = 'glorot_normal', activation = 'relu', input_dim = 81))
classifier.add(Dropout(p = 0.2))

# Adding the second hidden layer
#classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'glorot_normal', activation = 'relu'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 1, epochs = 1000, verbose=1)

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