import sys
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from rbf import RBF
from sklearn.metrics import mean_square_error
from numpy import mean
from random import random
import math

def hold_out(X):
    train, test = [], []
    for i in X: test.append(i) if int(random() * 4) == 0 else train.append(i)
    return (train, test)

read_data = lambda file_name: [map(float, line.split(' ')[0:-1]) for line in open(file_name,'r').readlines()]
data = lambda records: [record[0:-1] for record in records if len(record) == 9]
labels = lambda records: [record[-1] for record in records if len(record) == 9]
match_rate = lambda classifier, data, labels, Paser: mean_square_error(labels, [Paser(classifier.predict(i)) for i in data])
def best_grid(train_data, train_labels, validation_data, validation_labels, grid, Classifier, Paser):
    best = (sys.maxint, 0)
    for i in grid:
        classifier = Classifier(i)
        classifier.fit(train_data, train_labels)
        matches = match_rate(classifier, validation_data, validation_labels, Paser)
        if matches < best[0]: best = (matches, i)
    return best[1]

train, validation = hold_out(read_data('./data/bank8FM.data'))
train_data, train_labels = data(train), labels(train)
validation_data, validation_labels = data(validation), labels(validation)

test  = read_data('./data/bank8FM.test')
test_data, test_labels = data(test), labels(test)

radial_basis_functions_grid  = [1,2,5,10,20]
random_forest_regressor_grid = [2,3,4,5]
k_neighbors_regressor_grid   = [1,2,5,10,20]
svm_regressor_grid           = []
for i in range(-3, 6):
    for j in range(-3, 6): svm_regressor_grid.append((10**i,10**j))

print '------------------'
random_forest_regressor_best_grid = best_grid(
    train_data,
    train_labels,
    validation_data,
    validation_labels,
    random_forest_regressor_grid,
    lambda grid: RandomForestRegressor(max_features=grid),
    lambda result: mean(result)
)
print random_forest_regressor_best_grid
classifier = RandomForestRegressor(max_features=random_forest_regressor_best_grid)
classifier.fit(train_data + validation_data, train_labels + validation_labels)
print match_rate(classifier, test_data, test_labels, lambda result: mean(result))


print '------------------'
k_neighbors_regressor_best_grid = best_grid(
    train_data,
    train_labels,
    validation_data,
    validation_labels,
    k_neighbors_regressor_grid,
    lambda grid: KNeighborsRegressor(n_neighbors=grid),
    lambda result: mean(result)
)
print k_neighbors_regressor_best_grid
classifier = KNeighborsRegressor(n_neighbors=k_neighbors_regressor_best_grid)
classifier.fit(train_data + validation_data, train_labels + validation_labels)
print match_rate(classifier, test_data, test_labels, lambda result: mean(result))


print '------------------'
svm_regressor_best_grid = best_grid(
    train_data,
    train_labels,
    validation_data,
    validation_labels,
    svm_regressor_grid,
    lambda grid: SVR(C=grid[0], gamma=grid[1]),
    lambda result: result[0]
)
print svm_regressor_best_grid
classifier = SVR(C=svm_regressor_best_grid[0], gamma=svm_regressor_best_grid[1])
classifier.fit(train_data + validation_data, train_labels + validation_labels)
print match_rate(classifier, test_data, test_labels, lambda result: result[0])