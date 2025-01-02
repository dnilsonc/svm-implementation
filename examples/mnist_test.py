import sys
import os

# Adiciona o diretório raiz ao sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from src.svm import SVM # Certifique-se de que sua classe SVM está corretamente importada
from sklearn import svm

# 1. Carregar o Dataset
print("Carregando dataset...")
mnist = fetch_openml("mnist_784", version=1)
x, y = mnist.data, mnist.target

# 2. Filtrar para Apenas Duas Classes (ex.: dígitos 0 e 1)
print("Filtrando dataset...")
classes = ['3', '8']
filter_idx = np.isin(y, classes)
x, y = x[filter_idx], y[filter_idx]
y = np.where(y == '3', -1, 1)  # Converte para -1 e 1

# 3. Normalizar os Dados
print("Normalizando dataset...")
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

# 4. Dividir em Conjunto de Treinamento e Teste
print("Dividindo o dataset...")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=42)

# 5. Treinar o Modelo SVM
print("Treinando o modelo SVM...")
qp_svm = SVM()
qp_svm.fit(x_train, y_train)

# 6. Fazer Previsões
print("Realizando previsões...")
y_pred = qp_svm.predict(x_test)

# 7. Avaliação
accuracy = np.mean(y_pred.flatten() == y_test)
print(f"Acurácia: {accuracy * 100:.2f}%")

# ################################

# 8. Treinar o Modelo SVM scikit-learn
print("Treinando o modelo SVM scikit-learng...")
clf = svm.SVC(C=1, kernel='linear')
clf.fit(x_train, y_train)

# 6. Fazer Previsões
print("Realizando previsões...")
predict = clf.predict(x_test)

# 7. Avaliação
accuracy = np.mean(predict.flatten() == y_test)
print(f"Acurácia: {accuracy * 100:.2f}%")