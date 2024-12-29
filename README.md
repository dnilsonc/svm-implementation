# SVM - Implementação Simples

Esta é uma implementação simples de uma Máquina de Vetores de Suporte (Support Vector Machine - SVM) para classificação binária, utilizando a biblioteca `qpsolvers` para resolver problemas de otimização quadrática.

## Estrutura do Projeto

A classe `SVM` é responsável por treinar o modelo e realizar previsões.

### Métodos

#### `__init__(self)`
Inicializa a classe com os seguintes atributos:
- `self.weights`: Pesos do modelo após o treinamento.
- `self.bias`: Bias do modelo após o treinamento.

#### `fit(self, x_train, y_train)`
Treina o modelo SVM usando os dados de treino (`x_train`) e os rótulos (`y_train`).

**Parâmetros:**
- `x_train` (numpy array): Matriz \(N \times P\) contendo \(N\) amostras e \(P\) atributos.
- `y_train` (numpy array): Vetor de rótulos com \(N\) elementos (\(-1\) ou \(1\)).

**Funcionamento:**
1. Define a matriz \(P\) e o vetor \(q\) para a função objetivo.
2. Define as restrições \(G\) e \(h\) para o problema de otimização.
3. Resolve o problema de otimização utilizando `solve_qp`.
4. Armazena os pesos e o bias do modelo.

#### `predict(self, x_test)`
Realiza previsões para os dados de teste (`x_test`) com base no modelo treinado.

**Parâmetros:**
- `x_test` (numpy array): Matriz \(M \times P\) contendo \(M\) amostras e \(P\) atributos.

**Retorno:**
- `y_predict` (numpy array): Vetor de rótulos previstos (\(-1\) ou \(1\)) para cada amostra de teste.

## Dependências

- `numpy`
- `qpsolvers`

Instale as dependências com o comando:

```bash
pip install numpy qpsolvers
```

### Exemplo de Uso

```bash
import numpy as np

# Dados de treino
x_train = np.array([[1, 2], [2, 3], [3, 3]])
y_train = np.array([1, -1, 1])

# Inicializando e treinando o modelo
svm = SVM()
svm.fit(x_train, y_train)

# Dados de teste
x_test = np.array([[2, 2], [3, 4]])
y_pred = svm.predict(x_test)

print("Predições:", y_pred)
```