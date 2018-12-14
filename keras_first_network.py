#-------------------------------------------------------------------------------
#
# Keras First Neural Network
#
#-------------------------------------------------------------------------------

#!/usr/bin/env python
#!/path/to/interpreter

from keras.models import Sequential
from keras.layers import Dense
import numpy


# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# create model
model = Sequential()
# First layer: Input Variables
model.add(Dense(12, input_dim=8, activation='relu'))
# Second layer: Hidden Layer
model.add(Dense(8, activation='relu'))
# Third layer: Output Variable
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train/Fit the model
model.fit(X, Y, epochs=1500, batch_size=10)

# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
#predictions = model.predict(X)

# round predictions
#rounded = [round(x[0]) for x in predictions]
#print(rounded)
