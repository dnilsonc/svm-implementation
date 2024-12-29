import numpy as np
from qpsolvers import solve_qp
from scipy.sparse import csc_matrix

class SVM:
    """
    Implementação de uma Máquina de Vetores de Suporte (SVM - Support Vector Machine)
    utilizando Programação Quadrática para encontrar o hiperplano ótimo que separa duas classes.

    Atributos:
    ----------
    weights : np.ndarray
        Vetor de pesos (coeficientes) do hiperplano após o treinamento.
    bias : float
        Bias (termo independente) do hiperplano após o treinamento.
    C : float
        Parâmetro de regularização que controla o trade-off entre maximizar a margem e minimizar
        violações na classificação.

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
    """

    def __init__(self, C: float = 1.0):
        """
        Inicializa a classe SVM.

        Parâmetros:
        -----------
        C : float, opcional (default=1.0)
            Parâmetro de regularização que controla o trade-off entre margem e violações.
        """
        self.weights = None
        self.bias = None
        self.C = C

    def fit(self, x_train, y_train):
        """
        Treina o modelo SVM utilizando Programação Quadrática para encontrar o hiperplano ótimo.

        Parâmetros:
        -----------
        x_train : np.ndarray
            Matriz de treinamento (N x P), onde:
            - N é o número de amostras.
            - P é o número de atributos (features) por amostra.
        y_train : np.ndarray
            Vetor de rótulos das amostras de treinamento (N,).
            - Deve conter apenas valores -1 e 1.

        Detalhes:
        ---------
        Resolve o problema de otimização quadrática da forma:
            minimize    (1/2) * x.T @ P @ x + q.T @ x
            subject to  G @ x <= h
                        A @ x = b

        Após resolver, os pesos (`weights`) e o bias (`bias`) são extraídos da solução.
        """
        # Construção do problema dual
        B = x_train @ x_train.T  # Produto interno das amostras
        S = np.outer(y_train, y_train)  # Produto externo para aplicar restrições
        P = csc_matrix(S * B)  # Matriz de penalização
        q = -1 * np.ones(len(y_train))  # Vetor do problema dual

        # Restrições de igualdade
        A = csc_matrix(y_train[np.newaxis, :])  # Vetor para garantir a soma dos alphas
        b = np.zeros((1,))

        # Restrições de desigualdade (limites inferiores e superiores)
        lb = np.zeros(len(y_train))  # alpha >= 0
        lu = self.C * np.ones(len(y_train))  # alpha <= C

        # Solução do problema de Programação Quadrática
        alpha = solve_qp(P, q, None, None, A, b, lb, lu, solver="clarabel")

        # Cálculo dos pesos do modelo
        self.weights = np.sum((alpha * y_train)[:, np.newaxis] * x_train, axis=0)

        # Cálculo do bias usando todos os vetores de suporte
        self.bias = np.mean(y_train - np.sum(self.weights[np.newaxis, :] * x_train, axis=1))

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """
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
        """
        # Calcula a predição como o sinal da função de decisão
        y_predict = np.sign(x_test @ self.weights + self.bias)
        return y_predict