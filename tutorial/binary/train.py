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


# Fix random seed for reproducibility
numpy.random.seed(7)

# Load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# Split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# Use a Train-Test Split
# train, test = split(data)
# model = fit(train.X, train.y)
# predictions = model.predict(test.X)
# skill = compare(test.y, predictions)

# Create model: [8 input] -> [12 neurons] -> [1 output]
model = Sequential()
# Input Layer: Input Variables
model.add(Dense(12, input_dim=8, activation='relu'))
# First Hidden Layer
model.add(Dense(8, activation='relu'))
# Output Layer: Output Variables
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Model summary
#print(model.summary())
# Save model to a file
#plot_model(model, to_file='model.png')

# Train/Fit the model
history = model.fit(X, Y, validation_split=0.1, epochs=150, batch_size=10)

# Evaluate the model
scores = model.evaluate(X, Y)
# Loss
print("\n%s: %.2f" % (model.metrics_names[0], scores[0]))
# Accuracy
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# list all data in history
#print(history.history.keys())

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
