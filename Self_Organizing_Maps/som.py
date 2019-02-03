# Self Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:\\Users\\Louis\\Documents\\Computing\\AI\\Perso\\Deep_Learning_A_Z\\Self_Organizing_Maps\\Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
# The size of the SOM is arbitrary, we choose 10*10
# The input_len corresponds to the numbers of features of our dataset
# Sigma is distance of the different neighborhood in the grid
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
# Create the window
bone()
# Add the matrix of all the mean inter-neuron distances for all the wining nodes
# We take the transpose of this matrix with .T
pcolor(som.distance_map().T)
# Legend for the MID
# Frauds are rare and different from other operations, so they have bigger MID
# Because outline wining nodes correspond to the frauds, we can spot them with their high MID
colorbar()
# Add markers: red circles for customer who did not get banking card approval
# And green square for those who have been approved
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
	# Get the winning nodes
    w = som.winner(x)
    # Place the marker at the middle of the square
    # And add a marker according to its approval success
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds
mappings = som.win_map(X) # dict: coordinate on the matrix => list of costumer at this position
# Now we want to take the frauds identified on the graphed matrix (high MID) 
# And map their coordinate to their id thanks to the mappings variable
# TODO change the following coordinate to match the one of the MID nodes (look at the scale on the gui)
# Concacenate all the results
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)
# We scaled the values before, so now we unscalle them
frauds = sc.inverse_transform(frauds)
print(frauds)