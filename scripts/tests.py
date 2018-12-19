
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

# # load pima indians dataset
# dataset = numpy.loadtxt("tpcda19_02_dataset.csv", delimiter=",")
#
# # split into input (X) and output (Y) variables
# X = dataset[0,0:7]
# Y = dataset[0,7]
#
# print numpy.shape(dataset)
# print len(dataset)
#
# lines, columns = numpy.shape(dataset)
# print dataset[0:int(lines*0.5),0]
#
# # Node activation function
# f_acti = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh',
#           'sigmoid', 'hard_sigmoid', 'exponential', 'linear']
#
# for value in f_acti:
#     print value
#     #pass
#
# print "\n-----------------------------------------------------------------------"
# print "                            Results"
# #print "-----------------------------------------------------------------------"
#
# # Define the way to set the initial random weights and bias
# w_b_init = ['zeros', 'ones', 'random_normal','random_uniform']
# # Node activation function
# f_acti = ['sigmoid', 'elu', 'relu', 'linear']
#
# table = PrettyTable(["Index", "Layer weights", 'Layer bias', 'activation', "MSE"])
#
# i=0
# for w_init in w_b_init:
#   for b_init in w_b_init:
#     for f in f_acti:
#       table.add_row([i, w_init, b_init, f, 0])
#       i=i+1
#
# print table
#
#
# res = ["Top", "Layer weights", 'Layer bias', 'activation', "MSE"]
#
# res.remove("Top")
#
# print res


def main():

  # for value in range(11):
  #   print str(value/10.0*100) + '%'

  x = numpy.array([[40, 10], [50, 11]])

  print x

  _mean = x.mean(axis=1) # to take the mean of each row

  print _mean

  x.mean(axis=0)     # to take the mean of each col

  y_predictions = numpy.full((3,4),0)

  print y_predictions[0] # print one line

  y_predictions[0] = range(0,4)

  print y_predictions

  y_predictions_mean = y_predictions.mean(axis=0)

  print y_predictions_mean



# When Python interpreter reads a source file, it will execute all the code found in it
# When Python runs the "source file" as the main program, it sets the special variable
# (__name__) to have a value ("__main__")
# When you execute the main function, it will then read the "if" statement and
# checks whether __name__ does equal to __main__
# In Python "if__name__== "__main__" allows you to run the Python files either as
# reusable modules or standalone programs
if __name__== "__main__":
  main()
