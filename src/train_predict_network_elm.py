#-------------------------------------------------------------------------------
#
# TOP 10 with number of neurons on the hidden layer between 0 and 20;
#
#-------------------------------------------------------------------------------

#!/usr/bin/env python

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import optimizers
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.layers.advanced_activations import LeakyReLU, PReLU, ReLU
from sklearn.model_selection import train_test_split
import numpy
from keras import metrics
from prettytable import PrettyTable
import time


# Prepare data
def prepare_data():
  # Load and prepare the dataset
  dataset = numpy.loadtxt("../data/tpcda19_02_dataset.csv", delimiter=",")

  # Synthesis of the information contained in variables u6 and u7
  dataset[:,5] = 0.4878*dataset[:,5] + 0.5261*dataset[:,6]

  # Remove variable u2 from the dataset
  dataset = dataset[:,[0,2,3,4,5,7]]

  # Split into input (X) and output (y) variables
  X = dataset[:,0:5]
  y = dataset[:,5]
  return X, y

# ELM - acho que tens que iniciar assim, nao?
def setWeights(model):
		weights = model.layers[-1].get_weights()
		weights[0] = numpy.random.random_sample((weights[0].shape[0],weights[0].shape[1]))
		model.layers[-1].set_weights(weights)

# Define the advanced model: 5 inputs -> [ N ] -> 1 output
def build_network(hidden_nodes, u1, u2):
  # Create model
  model = Sequential()

  #Input Variables: Input Layer, Hidden Layer            FALTA: trainable=False ?
  if u1=='leaky_relu':
    model.add(Dense(hidden_nodes, input_dim=5, use_bias=False, trainable=False ))
    setWeights(model) # FALTA ISTO NAO?
    # add an advanced activation
    model.add(LeakyReLU(alpha=0.01)) #BEST: 0.01
  else:
    model.add(Dense(hidden_nodes, input_dim=5,
                    activation=u1, use_bias=False, trainable=False ))
    setWeights(model) # FALTA ISTO NAO?

  # Output layer: Output Variable
  if u2=='leaky_relu':
    model.add(Dense(1, use_bias=False ))
    # add an advanced activation
    model.add(LeakyReLU(alpha=0.00)) # BEST: 0.00
  else:
    model.add(Dense(1,
                    activation=u2, use_bias=False ))

  # Configures the model for a mean squared error regression problem
  model.compile(loss='mean_squared_error', optimizer='rmsprop')
  # sgd, adam, rmsprop, adadelta
  return model

# Network training
def train_network(model, X_train, y_train, iter, batch):
  # Trains the model for a given number of epochs (iterations on a dataset)
  history = model.fit(X_train, y_train, validation_split=0.10,
                      epochs=iter, batch_size=batch, verbose=1) #shuffle=False,
  return history

# Evaluate Network Mean Squared Error
def evaluate_network(model, X_test, y_test, batch):
  # Evaluate the model: Mean Squared Error (MSE)
  score = model.evaluate(X_test, y_test, batch_size=batch)
  return score

# Save TOP1 plot on a .png file
def save_plot_test_predection(y_test, y_predictions):
  print "\nSaving TOP1 Desired vs Predection plot..."
  # Plot desired values vs predictions
  plt.plot(y_test)
  plt.plot(y_predictions)
  plt.title('Desired vs Predection')
  plt.ylabel('y')
  plt.xlabel('X_test')
  plt.legend(['Desired', 'Predections'], loc='upper right')
  # ax = plt.gca()
  # Set x logaritmic
  # ax.set_yscale('log')
  plt.show()
  plt.savefig('../imgs/predictions_TOP1.png')

# Save TOP1 plot on a .png file
def save_mse_plot(history):
  print "\nSaving TOP1 MSE plots..."

  # Plot training & validation loss values
  plt.plot(history.history['loss'])
  # Validation set
  plt.plot(history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper right')
  # ax = plt.gca()
  # Set x logaritmic
  # ax.set_yscale('log')
  plt.show()
  plt.savefig('../imgs/TOP1_MSE.png')



# --------------------------------- MAIN ---------------------------------------
def main():

  # Load and prepare data
  X, y = prepare_data()

  # Split into Train dataset and Test data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

  # Fix random seed for reproducibility
  numpy.random.seed(100)

  # --------------------------- Paramenters ------------------------------------

  # Hidden node activation function
  u1 = 'leaky_relu'
  # Output node activation function
  u2 = 'relu' # better: leaky_relu > relu > linear > selu > elu
  # Initial weights - TOP sets (Test4)
  w1 = 'lecun_uniform' # TOP: lecun_uniform GOOD: random_normal,
  # Initial weights - TOP sets (Test4)
  w2 = 'VarianceScaling' # TOP: VarianceScaling GOOD: random_normal,
  # Initial bias - TOP sets (Test4)
  b = 'random_uniform'  # TOP: random_uniform GOOD: random_uniform

  # Create a Sequential model: [6 input] -> [12 neurons] -> [1 output]
  nodes = 20
  epochs = 1000 # 1000 epochs Matlab default != iterations
  batch = 32 # batch size

  model = build_network(nodes, u1, u2, w1, w2, b)
  time_start_train = time.clock()
  train_history = train_network(model, X_train, y_train, epochs, batch)
  time_elapsed_train = time.clock() - time_start
  time_start_train = time.clock()
  net_score = evaluate_network(model, X_test, y_test, batch)
  time_elapsed_test = time.clock() - time_start
  print(model.summary())
  #print(model.layers())

  # calculate predictions
  y_predictions = model.predict(X_test)
  #accuracy = numpy.mean(predictions == Y)

  print '\n\nMSE: ' + str(net_score)
  save_plot_test_predection(y_test, y_predictions)
  save_mse_plot(train_history)

  print 'Train Time: ', time_elapsed_train
  print 'Test Time: ', time_elapsed_test


if __name__== "__main__":
  main()
