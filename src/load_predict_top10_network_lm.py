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

# Define the advanced model: 5 inputs -> [ N ] -> 1 output
def build_network(hidden_nodes, u1, u2, w1, w2, b):
  # Create model
  model = Sequential()

  #Input Variables: Input Layer, Hidden Layer
  if u1=='leaky_relu':
    model.add(Dense(hidden_nodes, input_dim=5,
                      kernel_initializer=w1, bias_initializer=b,
                      use_bias=True ))
    # add an advanced activation
    model.add(LeakyReLU(alpha=0.01)) #BEST: 0.01
  else:
    model.add(Dense(hidden_nodes, input_dim=5,
                    activation=u1, kernel_initializer=w1, bias_initializer=b,
                    use_bias=True ))

  # Output layer: Output Variable
  if u2=='leaky_relu':
    model.add(Dense(1,
                    kernel_initializer=w2, bias_initializer=b,
                    use_bias=True ))
    # add an advanced activation
    model.add(LeakyReLU(alpha=0.00)) # BEST: 0.00
  else:
    model.add(Dense(1,
                    activation=u2, kernel_initializer=w2, bias_initializer=b,
                    use_bias=True ))

  # Configures the model for a mean squared error regression problem
  model.compile(loss='mean_squared_error', optimizer='adam')
  # sgd, adam, rmsprop, adadelta
  return model

# Network training
def train_network(model, X_train, y_train, iter):
  # Trains the model for a given number of epochs (iterations on a dataset)
  history = model.fit(X_train, y_train, validation_split=0.10,
                      epochs=iter, batch_size=32, verbose=0) #shuffle=False,
  return history

# Evaluate Network Mean Squared Error
def evaluate_network(model, X_test, y_test):
  # Evaluate the model: Mean Squared Error (MSE)
  score = model.evaluate(X_test, y_test, batch_size=32)
  print 'MSE: ' + str(score)
  return score

# Load TOP 10 models
def load_top10_models():
  for k in range(10):
    # load json and create model
    json_file = open('../models/model_TOP'+str(k+1)+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights('../models/model_TOP'+str(k)+'.h5')
  print("Loaded models from disk")
  return loaded_model

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
    plt.savefig('../imgs/ab/TOP'+str(k+1)+'_ab.png')


# --------------------------------- MAIN ---------------------------------------
def main():

  # Load and prepare data
  X, y = prepare_data()

  # Split into Train dataset and Test data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

  # Fix random seed for reproducibility
  numpy.random.seed(100)

  # --------------------------- Paramenters ------------------------------------
  # Define the way to set the initial random weights and bias
  #
  # zeros - initializer that generates tensors initialized to 0
  # ones - initializer that generates tensors initialized to 1
  # constant - "..." initialized to a constant value: (value=0)
  # random_normal - "..." normal distribution: (mean=0.0, stddev=0.05, seed=None)
  # random_uniform - "..." uniform distribution: (minval=-0.05, maxval=0.05, seed=None)
  # truncated_normal - "..." truncated normal distribution "...": (mean=0.0, stddev=0.05, seed=None)
  # VarianceScaling - "..." capable of adapting its scale to the shape of weights
  # orthogonal - "..." a random orthogonal matrix: (gain=1.0, seed=None)
  # Identity - "..." generates the identity matrix (Only use for 2D matrices): (gain=1.0)
  # lecun_uniform - LeCun uniform initializer: (seed=None)
  # glorot_normal - Glorot normal initializer, also called Xavier normal initializer: (seed=None)
  # glorot_uniform - Glorot uniform initializer, also called Xavier uniform initializer: (seed=None)
  # he_normal - He normal initializer: (seed=None)
  # lecun_normal: LeCun normal initializer: (seed=None)
  # he_uniform: He uniform variance scaling initializer: (seed=None)

  # Activation Functions
  #
  # softmax - softmax activation function: exp(z)/sum(exp(z))
  # softplus - softplus activation function: log(exp(x) + 1)
  # softsign - softsign activation function: x / (abs(x) + 1)
  # tanh - hyperbolic tangent activation function: tanh(x)
  # sigmoid - sigmoid activation function: 1 / (1 + exp(-x))
  # hard_sigmoid - hard sigmoid activation function: max(0, min(1, (x + 1)/2))
  # exponential - exponential (base e) activation function: exp(x)
  # elu - exponential linear unit: alfa*(exp(x) - 1),  x <= 0 | x, x > 0
  # selu - scaled Exponential Linear Unit (SELU): Y*alfa*(exp(x) - 1), Y*x < 0 | x, x >= 0
  # relu - rectified Linear Unit:  0,  x < 0 | x, x >= 0
  # linear - linear (i.e. identity) activation function: f(x) = x
  #
  # more complete resume:
  # - https://goo.gl/7hWj5y
  # sources:
  # - https://keras.io/activations/
  # - https://en.wikipedia.org/wiki/Activation_function
  #

  # TESTING: 3u1 x 1xu2 x 3w1 x 3w2 x 3b = 81 Networks
  # Hidden node activation function
  u_acti = ['tanh','relu','leaky_relu']
  # Output node activation function
  u2 = 'relu' # better: leaky_relu > relu > linear > selu > elu
  # Initial weights - TOP sets (Test4)
  w_init = ['truncated_normal','VarianceScaling','lecun_uniform']
  # Initial bias - TOP sets (Test4)
  b_init = ['zeros','random_uniform','random_normal']

  # Create a Sequential model: [6 input] -> [12 neurons] -> [1 output]
  #hidden_nodes = range(1,21) #0-20
  nodes = 20
  epochs = 1000 # 1000 epochs Matlab default != iterations
  model = []
  train_history = []
  net_score = []
  i=1
  # Analyses MSE for 6^(possible) config's of networks


  model = load_top10_models()


  for u1 in u_acti:
    for w1 in w_init:
      for w2 in w_init:
        for b in b_init:
          #for nodes in hidden_nodes:

            model.append(build_network(nodes, u1, u2, w1, w2, b))
            #train_history.append(train_network(model, X_train, y_train))
            train_network(model[i-1], X_train, y_train, epochs)
            net_score.append(evaluate_network(model[i-1], X_test, y_test))

            # Debug
            print "Model: ", u1, w1, w2, b, nodes
            print("Progress: %.2f%%" % (i/float(len(u_acti)
                                     *len(w_init)*len(w_init)*len(b_init)*1)*100))
            #print(model.summary())
            #print(model.layers())
            i=i+1


  top10, _index = find_top10score(net_score)

  table = find_top10config(net_score, top10, u_acti, w_init, b_init)
  save_top10(table)
  #save_top10plot(history, _index)


if __name__== "__main__":
  main()
