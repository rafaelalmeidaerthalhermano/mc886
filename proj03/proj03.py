from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.neighbors import NeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import random
import numpy as np

def print_result(rates):
    print int(np.mean(rates) * 100), '+/-', int(np.std(rates) * 100)

def fold(quantity, data):
    size = len(data[0]) / quantity
    for i in range(0, quantity):
        p     = i * size
        train = (data[0][0:p] + data[0][p + size:], data[1][0:p] + data[1][p + size:])
        test  = (data[0][p: p + size], data[1][p: p + size])
        yield (train, test)

def read_data(file_name):
    cast_int = lambda x: 1 if x == 'M' else 0
    lines    = open(file_name,'r').readlines()
    random.shuffle(lines)
    records  = [line[0:-1].split(',') for line in lines]
    data     = [map(float, record[0:-1]) for record in records]
    labels   = [cast_int(record[-1]) for record in records]
    return (data, labels)

def correct_predictions(classifier, test):
    matches = 0
    for i in range(len(test[0])):
        classified = classifier.predict(test[0][i])
        if classified[0] == test[1][i]: matches += 1
    return matches

def best_grid(data, grid, Classifier):
    best_k          = 0
    highest_matches = 0
    for K in grid:
        matches = 0
        for train, test in fold(5, data):
            classifier = Classifier(K)
            classifier.fit(train[0], train[1])
            matches += correct_predictions(classifier, test)
        if matches > highest_matches:
            best_k = K
            highest_matches = matches
    return best_k

def rates(data, grid, Classifier):
    rates = []
    for train, test in fold(5, data):
        classifier = Classifier(best_grid(train, grid, Classifier))
        classifier.fit(train[0], train[1])
        rate = correct_predictions(classifier, test) / float(len(test[0]))
        rates.append(rate)
    return rates

data = read_data("./data.csv")
data = (normalize(data[0]).tolist(), data[1])
pca  = PCA(n_components=12)
pca.fit(data[0])
data_pca = (pca.transform(data[0]).tolist(), data[1])

kneighbors_grid = [1,3,5,11,21,31]
kneighbors_rates = rates(data, kneighbors_grid, lambda K: NeighborsClassifier(n_neighbors=K))
kneighbors_pca_rates = rates(data_pca, kneighbors_grid, lambda K: NeighborsClassifier(n_neighbors=K))
print_result(kneighbors_rates)
print_result(kneighbors_pca_rates)

svm_linear_grid = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
svm_linear_rates = rates(data, svm_linear_grid, lambda C: SVC(C=C, kernel='linear'))
svm_linear_pca_rates = rates(data_pca, svm_linear_grid, lambda C: SVC(C=C, kernel='linear'))
print_result(svm_linear_rates)
print_result(svm_linear_pca_rates)

svm_rbf_grid    = [[0.001, 0.001],[0.01, 0.01],[0.1, 0.1],[1, 1],[10, 10],[100, 100],[1000, 1000],[10000, 10000],[0.001, 0.001],[0.01, 0.01],[0.1, 0.1],[1, 1],[10, 10],[100, 100],[1000, 1000],[10000, 10000],[0.001, 0.001],[0.01, 0.01],[0.1, 0.1],[1, 1],[10, 10],[100, 100],[1000, 1000],[10000, 10000],[0.001, 0.001],[0.01, 0.01],[0.1, 0.1],[1, 1],[10, 10],[100, 100],[1000, 1000],[10000, 10000],[0.001, 0.001],[0.01, 0.01],[0.1, 0.1],[1, 1],[10, 10],[100, 100],[1000, 1000],[10000, 10000],[0.001, 0.001],[0.01, 0.01],[0.1, 0.1],[1, 1],[10, 10],[100, 100],[1000, 1000],[10000, 10000],[0.001, 0.001],[0.01, 0.01],[0.1, 0.1],[1, 1],[10, 10],[100, 100],[1000, 1000],[10000, 10000],[0.001, 0.001],[0.01, 0.01],[0.1, 0.1],[1, 1],[10, 10],[100, 100],[1000, 1000],[10000, 10000]]
svm_rbf_rates = rates(data, svm_rbf_grid, lambda C: SVC(C=C[0], gamma=C[1], kernel='rbf'))
svm_rbf_pca_rates = rates(data_pca, svm_rbf_grid, lambda C: SVC(C=C[0], gamma=C[1], kernel='rbf'))
print_result(svm_rbf_rates)
print_result(svm_rbf_pca_rates)

randforest_grid = [2,3,5,10,20,40,60]
randforest_rates = rates(data, randforest_grid, lambda C: RandomForestClassifier(max_depth=C))
randforest_pca_rates = rates(data_pca, randforest_grid, lambda C: RandomForestClassifier(max_depth=C))
print_result(kneighbors_rates)
print_result(kneighbors_pca_rates)