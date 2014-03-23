import glob
from sklearn.decomposition import PCA
from sklearn.neighbors import NeighborsClassifier

def parse_data(file_name):
    file = open(file_name,'r').readlines()[3:]
    data = []
    for line in file:
        data = data + [int(element) for element in line.split(' ')[0:-1]]
    return data

def parse_class(file_name):
    return int(file_name.split('_')[0].split('/')[1])

def success_rate(classifier, test_data, test_class):
    matches = 0
    for i in range(0, len(test_data)):
        classified = classifier.predict(test_data[i])[0]
        if classified == test_class[i]:
            matches = matches + 1
    return matches

def test(train_data, test_data, train_class, test_class):
    neigh_1 = NeighborsClassifier(n_neighbors=1)
    neigh_1.fit(train_data, train_class)

    neigh_3 = NeighborsClassifier(n_neighbors=3)
    neigh_3.fit(train_data, train_class)

    neigh_5 = NeighborsClassifier(n_neighbors=5)
    neigh_5.fit(train_data, train_class)

    neigh_11 = NeighborsClassifier(n_neighbors=11)
    neigh_11.fit(train_data, train_class)

    neigh_17 = NeighborsClassifier(n_neighbors=17)
    neigh_17.fit(train_data, train_class)

    neigh_21 = NeighborsClassifier(n_neighbors=21)
    neigh_21.fit(train_data, train_class)

    print success_rate(neigh_1, test_data, test_class)
    print success_rate(neigh_3, test_data, test_class)
    print success_rate(neigh_5, test_data, test_class)
    print success_rate(neigh_11, test_data, test_class)
    print success_rate(neigh_17, test_data, test_class)
    print success_rate(neigh_21, test_data, test_class)

# CLASSES 1 and 7
print 'KNN with class 1 and 7'
train_files_17 = glob.glob("train17/*.pgm")
train_data_17  = [parse_data(file) for file in train_files_17]
train_class_17 = [parse_class(file) for file in train_files_17]
test_files_17 = glob.glob("test17/*.pgm")
test_data_17  = [parse_data(file) for file in test_files_17]
test_class_17 = [parse_class(file) for file in test_files_17]

print 'results without PCA'
test(train_data_17, test_data_17, train_class_17, test_class_17)

print 'results with 100 dimensions'
pca_17_100 = PCA(n_components=100)
pca_17_100.fit(train_data_17)
train_data_17_100 = pca_17_100.transform(train_data_17)
test_data_17_100  = pca_17_100.transform(test_data_17)
test(train_data_17_100, test_data_17_100, train_class_17, test_class_17)

print 'results with 40 dimensions'
pca_17_40 = PCA(n_components=40)
pca_17_40.fit(train_data_17)
train_data_17_40 = pca_17_40.transform(train_data_17)
test_data_17_40  = pca_17_40.transform(test_data_17)
test(train_data_17_40, test_data_17_40, train_class_17, test_class_17)

# CLASSES 4 and 9
print 'KNN with class 4 and 9'
train_files_49 = glob.glob("train49/*.pgm")
train_data_49  = [parse_data(file) for file in train_files_49]
train_class_49 = [parse_class(file) for file in train_files_49]
test_files_49 = glob.glob("test49/*.pgm")
test_data_49  = [parse_data(file) for file in test_files_49]
test_class_49 = [parse_class(file) for file in test_files_49]

print 'results without PCA'
test(train_data_49, test_data_49, train_class_49, test_class_49)

print 'results with 100 dimensions'
pca_49_100 = PCA(n_components=100)
pca_49_100.fit(train_data_49)
train_data_49_100 = pca_49_100.transform(train_data_49)
test_data_49_100  = pca_49_100.transform(test_data_49)
test(train_data_49_100, test_data_49_100, train_class_49, test_class_49)

print 'results with 40 dimensions'
pca_49_40 = PCA(n_components=40)
pca_49_40.fit(train_data_49)
train_data_49_40 = pca_49_40.transform(train_data_49)
test_data_49_40  = pca_49_40.transform(test_data_49)
test(train_data_49_40, test_data_49_40, train_class_49, test_class_49)