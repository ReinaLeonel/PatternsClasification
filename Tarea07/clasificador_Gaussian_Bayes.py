import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

data = pd.read_csv('Tarea07/Iris.csv')
print(data)
print(data['Species'].unique())

totalClases = len(data['Species'].unique())

clase1 = data[data['Species'] == 'Iris-setosa']
clase2 = data[data['Species'] == 'Iris-versicolor']
clase3 = data[data['Species'] == 'Iris-virginica']

print("Tamaño clase 1:", len(clase1))
print("Tamaño clase 2:", len(clase2))
print("Tamaño clase 3:", len(clase3))
print("Tamaño total:", len(data))

print("clase 1: ", clase1.head())
print("clase 2: ", clase2.head())
print("clase 3: ", clase3.head())

clasesData = [clase1, clase2, clase3]

# 1. Calculamos la probabilidad a priori de cada clase
def getPriorProb(lenData, clasesData):
    pW = []
    for clase in clasesData:
        pW.append(len(clase) / lenData)
    return pW

# 2. Calculamos la probabilidad condicional de cada clase (verosimilitud) calculando la media y la desviación estándar de cada atributo
def getMean_Std(clasesData):
    medias = []
    desviaciones = []
    for clase in clasesData:
        medias.append(clase.iloc[:, 1:5].mean().values)
        desviaciones.append(clase.iloc[:, 1:5].std().values)
    return medias, desviaciones

# 3. Se multiplican las probabilidades a priori por las verosimilitudes para cada clase y se elige la clase con mayor probabilidad posterior
def getGaussianProbability(x, mean, std):
    exponent = np.exp(-((x - mean) ** 2 / (2 * std ** 2)))
    fdp = (1 / (np.sqrt(2 * np.pi) * std)) * exponent
    return fdp

# 4
