from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GMM
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
import numpy as np
import math


data = [map(float, i[0:-1].split(' ')) for i in open('data','r').readlines()[0:-1]]

def intracluster_variance(data, clusters):
    size = len(data)
    sum  = 0
    for i in range(size):
        point   = data[i]
        cluster = clusters.labels_[i]
        center  = clusters.cluster_centers_[cluster]
        sum += np.linalg.norm(center - point)
    return math.sqrt(sum / size)

symbols = ['.', '+', '-', '*', 'x', 'o']

def print_cluster (cluster, data, k=6):
    norm_data = np.copy(data) / 5
    result    = []
    for i in range(k):
        result.append([])
        for j in range(len(norm_data)):
            if cluster[j] == i:
                result[i].append(norm_data[j])
    for i in range(len(result)):
        for j in result[i]:
            print "\draw (",j[0],",",j[1],") node[anchor=south] {",symbols[i],"};"

#GRID SEARCH
#for i in range(2,9):
#    cluster = KMeans(k=i)
#    cluster.fit(data)
#    print i, silhouette_score(data, labels=cluster.labels_), intracluster_variance(data, cluster)

# Se fossemos escolher apenas baseados na variancia intracluster, pegariamos k = 8, contudo, nesse caso, a silhueta fica
# abaixo de 5. portanto, escolhendo k = 6, encontramos a terceira menor variancia intracluster com silhueta maior do que
# 5, o que indica que uma estrutura razoavel foi encontrada.
km = 6
cluster_m = KMeans(k=2)
cluster_m.fit(data)
print_cluster(cluster_m.labels_, data)

#GM
#gmm = GMM(n_components=km)
#gmm.fit(data)
#print cluster_m.cluster_centers_ / 5
#print gmm.means  / 5

#single_cluster = fcluster(linkage(data, method='single'), 6, 'maxclust') - 1
#print_cluster(single_cluster, data)

#average_cluster = fcluster(linkage(data, method='average'), 6, 'maxclust') - 1
#print_cluster(average_cluster, data)

#complete_cluster = fcluster(linkage(data, method='complete'), 6, 'maxclust') - 1
#print_cluster(complete_cluster, data)
