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


# Load and prepare the dataset
dataset = numpy.loadtxt("tpcda19_02_dataset.csv", delimiter=",")

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

# # Define the way to set the initial random weights and bias
# w_b_init = ['normal', 'zeros', 'ones', 'random_normal','random_uniform',
#             'truncated_normal', 'lecun_uniform', 'lecun_normal', 'glorot_normal',
#             'glorot_uniform', 'he_normal', 'he_uniform']
# # Node activation function
# f_acti = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh',
#           'sigmoid', 'hard_sigmoid', 'exponential', 'linear']
# Define the way to set the initial random weights and bias
w_b_init = ['zeros', 'ones','random_normal','random_uniform']
# Node activation function
f_acti = ['sigmoid', 'elu', 'relu', 'linear']

# Create a Sequential model: [6 input] -> [12 neurons] -> [1 output]
input_nodes = 5
hidden_layer_nodes = 12
output_nodes = 1

score = []
j=1.0
for w_init in w_b_init:
  for b_init in w_b_init:
    for f in f_acti:
      model = Sequential()

      #Input Variables: Input Layer, Hidden Layer
      model.add(Dense(hidden_layer_nodes, input_dim=input_nodes,
                      kernel_initializer=w_init, bias_initializer=b_init,
                      activation=f ))

      # Output layer: Output Variable
      model.add(Dense(output_nodes,
                      kernel_initializer=w_init, bias_initializer=b_init ))

      # Configures the model for a mean squared error regression problem
      #model.compile(loss='mean_squared_error', optimizer='adam')
      model.compile(loss='mean_squared_error', optimizer='rmsprop')


      # Trains the model for a given number of epochs (iterations on a dataset)
      # Calculate the MSE for the training and test datasets generated
      # validation_data evaluate the loss and any model metrics at the end of each epoch
      # validation_data=(X_test, y_test)
      history = model.fit(X_train, y_train, validation_split=0.33,
                          epochs=500, batch_size=10, shuffle=True, verbose=0)


      # Evaluate the model: Mean Squared Error (MSE)
      score.append(model.evaluate(X_test, y_test, batch_size=10))

      # Loss
      #print("\n%s: %.6f" % (model.metrics_names[0], score))
      print "Model: ", w_init, b_init, f

      prog = (j/(len(w_b_init)*len(w_b_init)*len(f)))*100.0
      print "Progress: ", (j/64.0)*100
      j=j+1

#return

print "\n----------------------------------------------------------------------"
print "                                Top 10"

table = PrettyTable(["Top", "Layer weights", 'Layer bias', 'activation', "MSE"])

i=1
k=0
_score = score[:]
for w_init in w_b_init:
  for b_init in w_b_init:
    for f in f_acti:
      if score[k]==min(_score):
        table.add_row([i, w_init, b_init, f, min(_score)])
        _score.remove(min(_score))
        i=i+1
      k=k+1

print table


# calculate predictions
predictions = model.predict(X_train)

print "\nOutput: ", predictions[0]


# Plot training & validation loss values
plt.plot(history.history['loss'])
# Validation set
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
# ax = plt.gca()
# # Set x logaritmic
# ax.set_yscale('log')
plt.show()

#accuracy = metrics.binary_accuracy(float(Y), float(predictions))
#plt.plot(accuracy.eval())
