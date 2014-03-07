import numpy as np
import sys

file = []
for line in sys.stdin:
    file.append(line)

data  = [[float(row) for row in line[0:-1].split('\t')] for line in file[1:]]
atr_a = [record[0] for record in data]
atr_b = [record[1] for record in data]
atr_c = [record[2] for record in data]
atr_d = [record[3] for record in data]

hist_a_10 = np.histogram(atr_a, bins=10)
hist_a_30 = np.histogram(atr_a, bins=30)

hist_b_10 = np.histogram(atr_b, bins=10)
hist_b_30 = np.histogram(atr_b, bins=30)

hist_c_10 = np.histogram(atr_c, bins=10)
hist_c_30 = np.histogram(atr_c, bins=30)

hist_d_10 = np.histogram(atr_d, bins=10)
hist_d_30 = np.histogram(atr_d, bins=30)