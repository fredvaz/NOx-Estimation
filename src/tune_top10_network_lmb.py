#-------------------------------------------------------------------------------
#
# TOP 10 with number of neurons on the hidden layer between 0 and 20;
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

# Define the model: 5 inputs -> [ N ] -> 1 output
def build_network(hidden_layer_nodes, f1, f2, w1, b1, w2, b2):
  # Create model
  model = Sequential()

  #Input Variables: Input Layer, Hidden Layer
  model.add(Dense(hidden_layer_nodes, input_dim=5,
  activation=f1, kernel_initializer=w1, bias_initializer=b1, use_bias=True ))

  # Output layer: Output Variable
  model.add(Dense(1,
  activation=f2, kernel_initializer=w2, bias_initializer=b2, use_bias=True ))

  # Configures the model for a mean squared error regression problem
  model.compile(loss='mean_squared_error', optimizer='adam') #adam rmsprop

  return model

# Network training
def train_network(model, X_train, y_train):
  # Trains the model for a given number of epochs (iterations on a dataset)
  history = model.fit(X_train, y_train, validation_split=0.33,
                      epochs=150, batch_size=10, shuffle=False, verbose=0)
  return history

# Evaluate Network Mean Squared Error
def evaluate_network(model, X_test, y_test):
  # Evaluate the model: Mean Squared Error (MSE)
  score = model.evaluate(X_test, y_test, batch_size=10)
  print 'MSE: ' + str(score)
  return score

# Find Top 10 Score
def find_top10score(score):
  _score = score[:]
  top10 = []
  _index = []
  # Find Top 10 Score
  for k in range(10):
    top10.append(min(_score))
    _index.append(_score.index(min(_score)))
    _score.remove(min(_score))
  return top10, _index

# Find neural network config for Top 10
def find_top10config(score, top10, f_acti, w_init, b_init):
  table = PrettyTable(["Top", "w1", 'b1', 'w2', 'b2', 'f1', 'MSE'])
  k=0
  # Find neural network config for Top 10
  for k in range(10):
    i=0
    #in All measureed scores
    for f1 in f_acti:
      for w1 in w_init:
        for b1 in b_init:
          for w2 in w_init:
            for b2 in b_init:
              if score[i]==top10[k] and k < 10:
                table.add_row([k+1, f1, w1, b1, w2, b2, top10[k]])
              i=i+1
  return table

# Save TOP10 on a .txt file
def save_top10(table):
  print "\nSaving TOP 10 table..."
  data = table.get_string()
  with open('../data/TOP10_TABLE.txt', 'wb') as f:
    f.write(data)
  print table

# Save TOP10 plots on a .png file
def save_top10plot(history, _index):
  print "\nSaving TOP 10 MSE plots..."
  for k in range(10):
    # Plot training & validation loss values
    plt.plot(history[_index[k]].history['loss'])
    # Validation set
    plt.plot(history[_index[k]].history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    # ax = plt.gca()
    # # Set x logaritmic
    # ax.set_yscale('log')
    #plt.show()
    plt.savefig('../imgs/ab/TOP'+str(k)+'_ab.png')


# --------------------------------- MAIN ---------------------------------------
def main():

  # Load and prepare data
  X, y = prepare_data()

  # Split into Train dataset and Test data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

  # Fix random seed for reproducibility
  numpy.random.seed(5)

  # --------------------------- Paramenters ------------------------------------
  # Define the way to set the initial random weights and bias

  # Hidden node activation function
  f_acti = ['sigmoid', 'tanh', 'exponential', 'softplus'] #relu
  # Output node activation function
  f2 = 'linear'
  # Initial weights
  w_init = ['ones','random_normal','random_uniform'] # normal
  # Initial bias
  b_init = ['zeros','random_normal','random_uniform']

  # Create a Sequential model: [6 input] -> [12 neurons] -> [1 output]
  hidden_layer_nodes = 20 #12-20

  train_history = []
  net_score = []
  i=1
  # Analyses MSE for 6^(possible) config's of networks
  for f1 in f_acti:
    for w1 in w_init:
      for b1 in b_init:
        for w2 in w_init:
          for b2 in b_init:

            model = build_network(hidden_layer_nodes, f1, f2, w1, b1, w2, b2)
            #train_history.append(train_network(model, X_train, y_train))
            train_network(model, X_train, y_train)
            net_score.append(evaluate_network(model, X_test, y_test))

            # Debug
            print "Model: ", f1, w1, b1, w2, b2
            print("Progress: %.2f%%" % (i/float(len(f_acti)
                         *len(w_init)*len(b_init)*len(w_init)*len(b_init))*100))
            i=i+1


  top10, _index = find_top10score(net_score)
  table = find_top10config(net_score, top10, f_acti, w_init, b_init)
  save_top10(table)
  #save_top10plot(history, _index)


if __name__== "__main__":
  main()
