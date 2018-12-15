
# import numpy
#
# x = numpy.random.seed(7)
#
# print x
#
#
# # load pima indians dataset
# dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
#
# # split into input (X) and output (Y) variables
# X = dataset[:,0:8]
# Y = dataset[:,8]
#
# print X

import numpy

# load pima indians dataset
dataset = numpy.loadtxt("tpcda19_02_dataset.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[0,0:7]
Y = dataset[0,7]

print X
print Y
