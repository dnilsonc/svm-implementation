import numpy as np
from qpsolvers import solve_qp

class SVM:
    '''
    Classe para implementação de um classificador de Máquina de Vetores de Suporte (SVM - Support Vector Machine)
    utilizando Programação Quadrática para otimização.

    Atributos:
    ----------
    weights : np.ndarray
        Vetor de pesos (coeficientes) da hiperplano após o treinamento.
    bias : float
        Bias (termo independente) do hiperplano após o treinamento.

    Métodos:
    --------
    fit(x_train, y_train):
        Realiza o treinamento do modelo SVM utilizando os dados de entrada e suas respectivas classes.

    predict(x_test):
        Prediz as classes das amostras de teste com base no modelo treinado.

    Parâmetros esperados:
    ---------------------
    x_train : np.ndarray
        Matriz de treinamento com dimensão (N x P), onde:
        - N é o número de amostras.
        - P é o número de atributos (features) por amostra.
    y_train : np.ndarray
        Vetor de rótulos das classes das amostras de treinamento com dimensão (N,).
        - Deve conter valores -1 e 1, representando as duas classes.
    x_test : np.ndarray
        Matriz de dados de teste com dimensão (M x P), onde:
        - M é o número de amostras de teste.
        - P é o número de atributos (features) por amostra.
    '''

    def __init__(self):
        '''
        Inicializa a classe SVM.
        Define os pesos e o bias como None, pois serão definidos após o treinamento (fit).
        '''
        self.weights = None
        self.bias = None

    def fit(self, x_train, y_train):
        '''
        Treina o modelo SVM utilizando Programação Quadrática para encontrar o hiperplano ótimo.

        Parâmetros:
        -----------
        x_train : np.ndarray
            Matriz de treinamento (N x P), onde:
            - N é o número de amostras.
            - P é o número de atributos por amostra.
        y_train : np.ndarray
            Vetor de rótulos (N,) das amostras de treinamento.
            - Deve conter apenas valores -1 e 1.

        O algoritmo resolve o problema de otimização quadrática da seguinte forma:
            minimize    (1/2) * x.T @ P @ x + q.T @ x
            subject to  G @ x <= h

        Após resolver, os pesos (`weights`) e o bias (`bias`) são extraídos do vetor solução.
        '''
        # Matriz P define os pesos e desativa penalização do bias
        P = np.eye(x_train.shape[1] + 1); P[-1, -1] = 0  # Termo relacionado ao bias não deve ser penalizado

        # Vetor q (componente linear do problema de otimização)
        q = np.zeros(shape=(x_train.shape[1] + 1, 1))

        # Restrições do problema: G @ x <= h
        G = x_train * y_train[:, np.newaxis] 
        G = -1 * np.concatenate([G, y_train[:, np.newaxis]], axis=1) 
        h = -1 * np.ones(shape=(x_train.shape[0], 1))

        # Resolver o problema de programação quadrática
        x = solve_qp(P, q, G, h, solver="osqp")

        # Extração dos pesos e do bias da solução
        self.weights = x[:-1]
        self.bias = x[-1]

    def predict(self, x_test):
        '''
        Prediz as classes das amostras de teste com base no modelo treinado.

        Parâmetros:
        -----------
        x_test : np.ndarray
            Matriz de dados de teste (M x P), onde:
            - M é o número de amostras.
            - P é o número de atributos por amostra.

        Retorno:
        --------
        y_predict : np.ndarray
            Vetor de predições (M,), contendo os rótulos preditos para as amostras:
            -1 ou 1.
        '''
        y_predict = np.sign(x_test @ self.weights[:, np.newaxis] + self.bias)
        return y_predict
