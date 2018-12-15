#-------------------------------------------------------------------------------
#
# Keras First Neural Network
#
#-------------------------------------------------------------------------------

#!/usr/bin/env python
#!/path/to/interpreter

from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.utils import plot_model
import numpy
from keras import metrics


# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
#dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
dataset = numpy.loadtxt("tpcda19_02_dataset.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:7]
Y = dataset[:,7]

# load and prepare the dataset
# dataset = numpy.loadtxt("tpcda19_02_dataset.csv", delimiter=",")
# dataset[:,5] = 0.4878*dataset[:,5]+0.5261*dataset[:,6]
# dataset = dataset[:,[0,2,3,4,5,7]]
# X = dataset[0:2000,0:6]
# X_test = dataset[2000:2199,0:6]
# Y = dataset[0:2000,6]
# Y_test = dataset[2000:2199,6]

# create model
model = Sequential()
# First Hidden Layer: Input Variables
model.add(Dense(10, input_dim=7, kernel_initializer='normal', activation='elu'))
# Second Hidden Layer
#model.add(Dense(8, activation='relu'))
# Third layer: Output Variable
model.add(Dense(1, kernel_initializer='normal'))

# Configures the model for training
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mean_squared_error', optimizer='adam')#,
              #metrics=[metrics.binary_accuracy])

# Train/Fit the model
history = model.fit(X, Y, epochs=1000, batch_size=10) #, verbose=0

# evaluate the model
scores = model.evaluate(X, Y)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
predictions = model.predict(X)

print predictions[1:5]

#accuracy = metrics.binary_accuracy(float(Y), float(predictions))
#plt.plot(accuracy.eval())
#print dataset[1,0:2], dataset[1,7]



# Plot training & validation accuracy values
plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Save it to a file
#plot_model(model, to_file='model.png')


# round predictions
#rounded = [round(x[0]) for x in predictions]
#print(rounded)
