#!/usr/bin/env python
#!/path/to/interpreter

import numpy


# Calculate predictions
predictions = model.predict(X)

# Round predictions
#rounded = [round(x[0]) for x in predictions]
#print(rounded)
print "Some predictions: ", predictions[1:5]

# 5. make predictions
probabilities = model.predict(X)
predictions = [float(round(x)) for x in probabilities]
accuracy = numpy.mean(predictions == Y)
print("Prediction Accuracy: %.2f%%" % (accuracy*100))
