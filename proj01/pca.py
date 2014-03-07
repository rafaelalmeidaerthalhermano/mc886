from sklearn.decomposition import PCA
import sys

file = []
for line in sys.stdin:
    file.append(line)

data = [[float(row) for row in line[0:-1].split('\t')] for line in file[1:]]

pca = PCA(n_components=2)
new_data = pca.fit_transform(data)

for i in range(0, len(new_data)):
    print "\draw (", new_data[i][0], ',', new_data[i][1] ,") node[anchor=south] {.};"