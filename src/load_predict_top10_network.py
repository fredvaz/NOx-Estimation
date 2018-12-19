#-------------------------------------------------------------------------------
#
# TOP 10 with number of neurons on the hidden layer between 0 and 20;
#
#-------------------------------------------------------------------------------

#!/usr/bin/env python

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers.advanced_activations import LeakyReLU, PReLU, ReLU
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
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

# Evaluate Network Mean Squared Error
def evaluate_network(model, X_test, y_test):
  # Evaluate the model: Mean Squared Error (MSE)
  score = model.evaluate(X_test, y_test, batch_size=32)
  print 'MSE: ' + str(score)
  return score

# Load TOP 10 models
def load_top10_models(k, set):
  # load json and create model
  json_file = open('../models/'+set+'/model_TOP'+str(k+1)+'.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  # load weights into new model
  loaded_model.load_weights('../models/'+set+'/model_TOP'+str(k+1)+'.h5')
  print '\nLoaded TOP'+str(k+1)+' model from disk'
  return loaded_model

# Save TOP1 plot on a .png file
def save_plot_top_test_predection(y_test, y_predictions, _str, k):
  print "Saving TOP Desired vs Predection plot..."
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
  #plt.show()
  plt.savefig('../imgs/predections_'+_str+'/predictions_TOP'+str(k)+'.png')

def save_plot_mean_test_predection(y_test, y_predictions, _str, k):
  print "Saving TOP Desired vs Predection plot..."
  # Plot desired values vs predictions
  plt.plot(y_test)
  plt.plot(y_predictions)
  plt.title('Desired vs Predection TOP 10 MEAN')
  plt.ylabel('y mean')
  plt.xlabel('X_test')
  plt.legend(['Desired', 'Predections'], loc='upper right')
  # ax = plt.gca()
  # Set x logaritmic
  # ax.set_yscale('log')
  plt.show()
  #plt.savefig('../imgs/predections_'+_str+'/predictions_TOP'+str(k)+'.png')


# --------------------------------- MAIN ---------------------------------------
def main():

  # Load and prepare data
  X, y = prepare_data()

  # Split into Train dataset and Test data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

  # Fix random seed for reproducibility
  numpy.random.seed(100)

  # ------------------------------- TEST --------------------------------------
  net_score = []
  time_elapsed_test = []
  y_predictions = numpy.full((len(y_test),0),0)
  # dataset != dataset treino:
  for k in range(0,10):
    model = load_top10_models(k, 'test_100_epochs')
    # Configures the model for a mean squared error regression problem
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    #print(model.layers())
    #print(model.summary())
    # MSE
    score = evaluate_network(model, X_test, y_test)
    # calculate predictions
    time_start_test = time.clock()
    _y = model.predict(X_test)
    time_elapsed_test.append(time.clock() - time_start_test)
    #save_plot_top_test_predection(y_test, _y, 'train', k)
    net_score.append(score)
    y_predictions = numpy.hstack([y_predictions, _y])
  # Score Mean TOP10
  net_score_mean = numpy.mean(net_score)
  print '\n\nTEST MSE MEAN: ' + str(net_score_mean) +'\n\n'
  # Desired vs Predection TOP10 Mean
  y_predictions_mean = y_predictions.mean(axis=1) # to take the mean of each row

  # Plot Mean
  save_plot_mean_test_predection(y_test, y_predictions_mean, 'test_mean', k)

  print 'Mean Time Test: ', numpy.mean(time_elapsed_test)


  # ------------------------------- TRAIN --------------------------------------
  net_score = []
  time_elapsed_train = []
  #y_predictions = []
  y_predictions = numpy.full((len(y_train),0),0)
  # dataset = dataset treino:
  for k in range(0,10):
    model = load_top10_models(k, 'test_100_epochs')
    # Configures the model for a mean squared error regression problem
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    #print(model.layers())
    #print(model.summary())
    # MSE
    score = evaluate_network(model, X_train, y_train)
    # calculate predictions
    time_start_train = time.clock()
    _y = model.predict(X_train)
    time_elapsed_train.append(time.clock() - time_start_train)
    #save_plot_top_test_predection(y_train, _y, 'train', k)
    net_score.append(score)
    y_predictions = numpy.hstack([y_predictions, _y])

  # Score Mean TOP10
  net_score_mean = numpy.mean(net_score)
  print '\n\nTRAIN MSE MEAN: ' + str(net_score_mean) +'\n\n'
  # Desired vs Predection TOP10 Mean
  y_predictions_mean = y_predictions.mean(axis=1) # to take the mean of each row

  # Plot Mean
  save_plot_mean_test_predection(y_train, y_predictions_mean, 'train_mean', k)

  print 'Mean Time Train: ', numpy.mean(time_elapsed_train), '\n\n'


if __name__== "__main__":
  main()
