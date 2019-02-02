# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Installing Pillow
# pip install pillow

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
# 32 features vector size 3*3 and reshape input images to 64*64 and 3 colors (RGB)
# The input size of the images should be use again later for train_datagen with the same size
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
# Classic 2*2 max pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
# Because we are on a second convolutionnal layer, we can increase (double for example) the number of feature vectors
# This is because the deeper we go, the more precise the features detected in the image can be
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
# We need to choose a lower number of inputs neurons that the size of the flattened vector because it is huge
# But we need not to reduce the size too much because we will loose accuracy
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
# As it is only a two categories classification we choose the binary_crossentropy loss
# If we have more than two, we need the categorical_crossentropy loss
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

# Generate batches of tensor image data with real-time data augmentation
# This create small changes in our images to reduce overfitting
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# We use the ImageDataGenerator to rescale it to a fixed size
test_datagen = ImageDataGenerator(rescale = 1./255)

# These two sections create the training set and the test set
# We the use the image size specified in the first activation function for the target_size
training_set = train_datagen.flow_from_directory('C:\\Users\\Louis\\Documents\\Computing\\AI\\Perso\\Deep_Learning_A_Z\\Convolutional_Neural_Networks\\dataset\\training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('C:\\Users\\Louis\\Documents\\Computing\\AI\\Perso\\Deep_Learning_A_Z\\Convolutional_Neural_Networks\\dataset\\test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)