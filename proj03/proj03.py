from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedKFold
from sklearn.neighbors import NeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def print_result(rates):
    print int(np.mean(rates) * 100), '+/-', int(np.std(rates) * 100)

def read_data(file_name):
    cast_int = lambda x: 1 if x == 'M' else 0
    lines    = open(file_name,'r').readlines()
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
        for train_index, test_index in StratifiedKFold(data[1], 5):
            train = (
                [i for j, i in enumerate(data[0]) if j not in train_index],
                [i for j, i in enumerate(data[1]) if j not in train_index]
            )
            test  = (
                [i for j, i in enumerate(data[0]) if j not in test_index],
                [i for j, i in enumerate(data[1]) if j not in test_index]
            )
            classifier = Classifier(K)
            classifier.fit(train[0], train[1])
            matches += correct_predictions(classifier, test)
        if matches > highest_matches:
            best_k = K
            highest_matches = matches
    return best_k

def rates(data, grid, Classifier):
    rates    = []
    for train_index, test_index in StratifiedKFold(data[1], 5):
        train          = (
            [i for j, i in enumerate(data[0]) if j not in train_index],
            [i for j, i in enumerate(data[1]) if j not in train_index]
        )
        test           = (
            [i for j, i in enumerate(data[0]) if j not in test_index],
            [i for j, i in enumerate(data[1]) if j not in test_index]
        )
        hiperparameter = best_grid(train, grid, Classifier)
        classifier     = Classifier(hiperparameter)
        classifier.fit(train[0], train[1])
        rate = correct_predictions(classifier, test) / float(len(test[0]))
        rates.append(rate)
    return rates


data = read_data("./data.csv")
data = (normalize(data[0]).tolist(), data[1])

data_pca = read_data("./data.csv")
data_pca = (normalize(data_pca[0]).tolist(), data_pca[1])

pca  = PCA()
pca.fit(data_pca[0])
data_pca = (pca.transform(data_pca[0]).tolist(), data_pca[1])

kneighbors_grid = [1,3,5,11,21,31]
svm_linear_grid = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
svm_rbf_grid = []
for i in svm_linear_grid:
    for j in svm_linear_grid:
         svm_rbf_grid.append([i,j])
randforest_grid = [2,3,5,10,20,40,60]

kneighbors_rates = rates(data, kneighbors_grid, lambda K: NeighborsClassifier(n_neighbors=K))
print_result(kneighbors_rates)
kneighbors_pca_rates = rates(data_pca, kneighbors_grid, lambda K: NeighborsClassifier(n_neighbors=K))
print_result(kneighbors_pca_rates)

svm_linear_rates = rates(data, svm_linear_grid, lambda C: SVC(C=C, kernel='linear'))
print_result(svm_linear_rates)
svm_linear_pca_rates = rates(data_pca, svm_linear_grid, lambda C: SVC(C=C, kernel='linear'))
print_result(svm_linear_pca_rates)

svm_rbf_rates = rates(data, svm_rbf_grid, lambda C: SVC(C=C[0], gamma=C[1], kernel='rbf'))
print_result(svm_rbf_rates)
svm_rbf_pca_rates = rates(data_pca, svm_rbf_grid, lambda C: SVC(C=C[0], gamma=C[1], kernel='rbf'))
print_result(svm_rbf_pca_rates)

randforest_rates = rates(data, randforest_grid, lambda C: RandomForestClassifier(max_depth=C))
print_result(kneighbors_rates)
randforest_pca_rates = rates(data_pca, randforest_grid, lambda C: RandomForestClassifier(max_depth=C))
print_result(kneighbors_pca_rates)