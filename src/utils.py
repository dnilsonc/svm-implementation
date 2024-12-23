import numpy as np

def accuracy(y_test, y_predict):
    """
    Calcula a acurácia como a proporção de previsões corretas.

    Args:
        y_test (np.ndarray): Rótulos reais.
        y_predict (np.ndarray): Rótulos previstos.

    Returns:
        float: Proporção de acertos (valor entre 0 e 1).

    Raises:
        ValueError: Se `y_test` e `y_predict` não tiverem o mesmo número de elementos.
    """
    if len(y_test.flat) != len(y_predict.flat):
        raise ValueError("y_test e y_predict devem ter o mesmo número de elementos.")

    return np.sum(y_test.flat == y_predict.flat) / len(y_predict.flat)
