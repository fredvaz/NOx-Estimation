#-------------------------------------------------------------------------------
#
# Analyses MSE for TOP 1 config's of networks
#
#-------------------------------------------------------------------------------

#!/usr/bin/env python

from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
import numpy
from keras import metrics
from prettytable import PrettyTable


# Load and prepare the dataset
dataset = numpy.loadtxt("../data/tpcda19_02_dataset.csv", delimiter=",")

# Synthesis of the information contained in variables u6 and u7
dataset[:,5] = 0.4878*dataset[:,5] + 0.5261*dataset[:,6]

# Remove variable u2 from the dataset
dataset = dataset[:,[0,2,3,4,5,7]]

# Split into input (X) and output (y) variables
X = dataset[:,0:5]
y = dataset[:,5]

# Split into Train dataset and Test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25);


# Fix random seed for reproducibility
numpy.random.seed(5)


# Define the BEST config:

# Initial random weights and bias
w1 = 'random_normal'
b1 = 'random_normal'
w2 = 'random_normal'
b2 = 'random_normal'
# Activation function
f1 = 'linear'
f2 = 'linear'


# Create a Sequential model: [6 input] -> [12 neurons] -> [1 output]
model = Sequential()

#Input Variables: Input Layer, Hidden Layer
model.add(Dense(12, input_dim=5,
                kernel_initializer=w1, bias_initializer=b1, activation=f1 ))

# Output layer: Output Variable
model.add(Dense(1,
                kernel_initializer=w2, bias_initializer=b2, activation=f2 ))


# Configures the model for a mean squared error regression problem
model.compile(loss='mean_squared_error', optimizer='rmsprop') #adam


# Trains the model for a given number of epochs (iterations on a dataset)
history = model.fit(X_train, y_train, validation_split=0.33,
                    epochs=150, batch_size=10, shuffle=True, verbose=1)


# Evaluate the model: Mean Squared Error (MSE)
score = model.evaluate(X_test, y_test, batch_size=10)

# Loss
print("\nMSE: %.6f" % score)

print "\n---- Desired vs Predections ----"

# calculate predictions
y_predictions = model.predict(X_test)

# Plot desired values vs predictions
plt.plot(y_test)
plt.plot(y_predictions)
plt.title('Desired vs Predections')
plt.ylabel('y')
plt.xlabel('Epoch')
plt.legend(['Desired', 'Predections'], loc='upper right')
# ax = plt.gca()
# # Set x logaritmic
# ax.set_yscale('log')
plt.show()
plt.savefig('../imgs/predictions_top1.png')
