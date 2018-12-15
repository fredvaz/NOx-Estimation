#-------------------------------------------------------------------------------
#
# Regression Problem
# scikit-learn with Keras to evaluate models
# - Cross-validation (statistics):
#   a technique for estimating the performance of a predictive model
# cross-validation combines (averages) measures of fitness in prediction to
# derive a more accurate estimate of model prediction performance
#
# tune the network topology of models with Keras
#
#-------------------------------------------------------------------------------

#!/usr/bin/env python
#!/path/to/interpreter

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# evaluate model with standardized dataset
# def standardize_dataset(_model, X, Y):
#     estimators = []
#     estimators.append(('standardize', StandardScaler()))
#
#     kfold = KFold(n_splits=10, random_state=seed)
#     results = cross_val_score(estimator, X, Y, cv=kfold)
#     print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

# define the model: 13 inputs -> [13 -> 6] -> 1 output
def larger_model():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

# define wider model: 13 inputs -> [20] -> 1 output
def wider_model():
	# create model
	model = Sequential()
	model.add(Dense(20, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

# train and evaluate model
def train_model(_model, seed, X, Y):

    # train model
    estimator = KerasRegressor(build_fn=_model, epochs=100, batch_size=5) # verbose=0

    # evaluate across 10 folds of the cross validation evaluation
    kfold = KFold(n_splits=10, random_state=seed)
    results = cross_val_score(estimator, X, Y, cv=kfold)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# train and evaluate model with standardized dataset
def train_model_standardized(_model, seed, X, Y):

    # train model
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=_model, epochs=50, batch_size=5))) # verbose=0

    # evaluate across 10 folds of the cross validation evaluation
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=10, random_state=seed)
    results = cross_val_score(pipeline, X, Y, cv=kfold)
    print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))


def main():
    # my code here

    # load dataset
    print "loading data..."
    dataframe = pandas.read_csv("housing.csv", delim_whitespace=True, header=None)
    dataset = dataframe.values

    # split into input (X) and output (Y) variables
    print "spliting data..."
    X = dataset[:,0:13]
    Y = dataset[:,13]

    #print X[0:5], Y[0]
    #print "standardize data..."
    #X, Y = standardize_dataset(baseline_model, X, Y)

    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)

    print "training network..."
    #_model = baseline_model()
    #train_model(_model, seed, X, Y)
    #train_model_standardized(_model, seed, X, Y)

    # create model
    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')

    # evaluate model with standardized dataset
    # estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5) #verbose=0
    # kfold = KFold(n_splits=10, random_state=seed)
    # results = cross_val_score(estimator, X, Y, cv=kfold)
    # print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

    # estimators = []
    #
    # estimators.append(('standardize', StandardScaler()))

    # Train/Fit the model
    model.fit(X, Y, epochs=100, batch_size=5)

    # evaluate the model
    #scores = model.evaluate(X, Y)



    # calculate predictions
    predictions = model.predict(X)
    print predictions[1:5]





if __name__ == "__main__":
    main()
