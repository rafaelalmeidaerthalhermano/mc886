import numpy as np
import sys

file = []
for line in sys.stdin:
    file.append(line)

data = [[float(row) for row in line[0:-1].split('\t')] for line in file[1:]]

print np.cov(data, rowvar=None)