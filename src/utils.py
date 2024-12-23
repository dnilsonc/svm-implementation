import numpy as np

def accuracy(y_test, y_predict):
    return len((y_test.flat == y_predict.flat)) / len(y_predict.flat)
