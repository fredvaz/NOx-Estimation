#-------------------------------------------------------------------------------
#
# Keras First Neural Network
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


def gen():
    while True:
        yield X_train, y_train


# Load and prepare the dataset
dataset = numpy.loadtxt("../../data/tpcda19_02_dataset.csv", delimiter=",")

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


# SET: 3 x 3 x 11 x 11 = 1089

# Define the way to set the initial random weights and bias
w_b_init = ['glorot_uniform', 'he_normal', 'he_uniform']
# Node activation function
f_acti = ['softmax', 'elu', 'selu']


# Create a Sequential model: [6 input] -> [12 neurons] -> [1 output]
hidden_layer_nodes = 12

history = []
score = []
j=1.0
# Analyses MSE for 6^(options) config's of networks
for w1 in w_b_init:
  for b1 in w_b_init:
    for w2 in w_b_init:
      for b2 in w_b_init:
        for f1 in f_acti:
          for f2 in f_acti:
            model = Sequential()

            #Input Variables: Input Layer, Hidden Layer
            model.add(Dense(hidden_layer_nodes, input_dim=5,
                            kernel_initializer=w1, bias_initializer=b1, activation=f1 ))

            # Output layer: Output Variable
            model.add(Dense(1,
                            kernel_initializer=w2, bias_initializer=b2, activation=f2 ))

            # Configures the model for a mean squared error regression problem
            model.compile(loss='mean_squared_error', optimizer='rmsprop') #adam

            # Trains the model for a given number of epochs (iterations on a dataset)
            history.append(model.fit(X_train, y_train, validation_split=0.33,
                                epochs=150, batch_size=10, shuffle=True, verbose=0))

            # Evaluate the model: Mean Squared Error (MSE)
            score.append(model.evaluate(X_test, y_test, batch_size=10))

            # Loss
            #print("\n%s: %.6f" % (model.metrics_names[0], score))
            print "Model: ", w1, b1, w2, b2, f1, f2

            # prog = (j/(len(w_b_init)*len(w_b_init)*len(f)))*100.0
            # print "Progress: ", (j/float(len(w_b_init)*len(w_b_init)*len(f)))*100
            # j=j+1
#return
print "\n---------------------------------------------------------------------------"
print "                                TOP 10"
table = PrettyTable(["Top", "w1", 'b1', 'w2', 'b2', 'f1', 'f2', 'MSE'])
_score = score[:]
top10 = []
_index = []
# Find Top 10 Score
for k in range(10):
  top10.append(min(_score))
  _index.append(index(min(_score)))
  _score.remove(min(_score))

i=0
# Find neural network config for Top 10
for k in range(10):
  for w1 in w_b_init:
    for b1 in w_b_init:
      for w2 in w_b_init:
        for b2 in w_b_init:
          for f1 in f_acti:
            for f2 in f_acti:
              if score[i]==min(top10):
                table.add_row([k+1, w1, b1, w2, b2, f1, f2, min(top10)])
                top10.remove(min(top10))
              i=i+1
print table
print "\n ------ Save TOP 10 --------"
data = x.get_string()
with open('../../data/TOP10_TABLE_da.txt', 'wb') as f:
  f.write(data)

print "\n---- Save Plot TOP 10 ------"
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
  plt.savefig(['../../imgs/TOP'], k, ['_da.png'])


#if __name__ == "__main__":
