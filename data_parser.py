import csv
import numpy as np


def read(filename='data.csv'):
    ret = []
    with open('data.csv', mode='r') as f:
        reader = csv.reader(f)
        for row in reader:
            ret.append(row)
    return np.array(ret, dtype=np.float64)
