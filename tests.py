
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
from tabulate import tabulate
from prettytable import PrettyTable

# load pima indians dataset
dataset = numpy.loadtxt("tpcda19_02_dataset.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[0,0:7]
Y = dataset[0,7]

print numpy.shape(dataset)
print len(dataset)

lines, columns = numpy.shape(dataset)
print dataset[0:int(lines*0.5),0]

# Node activation function
f_acti = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh',
          'sigmoid', 'hard_sigmoid', 'exponential', 'linear']

for value in f_acti:
    print value
    #pass

print "\n-----------------------------------------------------------------------"
print "                            Results"
#print "-----------------------------------------------------------------------"

# Define the way to set the initial random weights and bias
w_b_init = ['zeros', 'ones', 'random_normal','random_uniform']
# Node activation function
f_acti = ['sigmoid', 'elu', 'relu', 'linear']

table = PrettyTable(["Index", "Layer weights", 'Layer bias', 'activation', "MSE"])

i=0
for w_init in w_b_init:
  for b_init in w_b_init:
    for f in f_acti:
      table.add_row([i, w_init, b_init, f, 0])
      i=i+1

print table


res = ["Top", "Layer weights", 'Layer bias', 'activation', "MSE"]

res.remove("Top")

print res
