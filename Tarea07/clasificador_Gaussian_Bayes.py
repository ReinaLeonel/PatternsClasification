import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

data = pd.read_csv('Tarea07/Iris.csv')
print(data)
print(data['Species'].unique())

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


# 1. Calculamos la probabilidad a priori de cada clase
pAPriori = []
clasesData = [clase1, clase2, clase3]
for clase in clasesData:
    pAPriori.append(len(clase) / len(data))

