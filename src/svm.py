import numpy as np
from qpsolvers import solve_qp

class SVM:
    '''
    x_train: N x P (Numero de amostras x Atributos)
    '''
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, x_train, y_train):   
        P = np.eye(x_train.shape[1] + 1); P[-1, -1] = 0
        q = np.zeros(shape=(x_train.shape[1] + 1, 1))

        G = x_train * y_train[:, np.newaxis]
        G = np.concat([G, y_train[:, np.newaxis]], axis=1)
        h = -1 * np.ones(shape=(x_train.shape[0], 1))

        x = solve_qp(P, q, G, h, solver="cvxopt")

        self.weights = x[:-1]
        self.bias = x[-1]

    def predict(self, x_test):
        y_predict = np.sign(x_test @ self.weights[:, np.newaxis] + self.bias)
        return y_predict