import pandas as pd
import numpy as np
import math
from sklearn.model_selection import KFold

# data = pd.read_csv('Tarea07/Iris.csv')
# print(data)
# print(data['Species'].unique())

# totalClases = len(data['Species'].unique())

# clase1 = data[data['Species'] == 'Iris-setosa']
# clase2 = data[data['Species'] == 'Iris-versicolor']
# clase3 = data[data['Species'] == 'Iris-virginica']

# print("Tamaño clase 1:", len(clase1))
# print("Tamaño clase 2:", len(clase2))
# print("Tamaño clase 3:", len(clase3))
# print("Tamaño total:", len(data))

# print("clase 1: ", clase1.head())
# print("clase 2: ", clase2.head())
# print("clase 3: ", clase3.head())

# clasesData = [clase1, clase2, clase3]

def readData(rutaArchivo):
    data = pd.read_csv(rutaArchivo)
    return data

def getClasesData(data):
    clasesData = []
    lenClases = len(data['Species'].unique())
    for clase in data['Species'].unique():
        clasesData.append(data[data['Species'] == clase])
    return clasesData, lenClases

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
    # print("Calculando probabilidad Gaussiana para x:", x, "mean:", mean, "std:", std)
    elevacion = -((x - mean) ** 2 / (2 * std ** 2))
    elevacion = np.array(elevacion, dtype=np.float64)
    # print("Elevacion:", elevacion)
    # print("Type elevacion:", type(elevacion))
    # print(elevacion.dtype)
    # print(elevacion)
    exponent = np.exp(elevacion)
    # print("Exponent:", exponent)
    fdp = (1 / (np.sqrt(2 * np.pi) * std)) * exponent
    return fdp



# Main
data = readData('./Tarea07/Iris.csv')
clasesData, totalClases = getClasesData(data)

print("Tamaño total de datos:", len(data))
print("Total de clases:", totalClases)
# print(clasesData[2])

# 1. Calculamos la probabilidad a priori de cada clase
pW = getPriorProb(len(data), clasesData)
print("Probabilidades a priori de cada clase:", pW)

# 2. Dividimos los datos en K-Folds
kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold = 1
for train_index, test_index in kf.split(data):
    print("=======================================")
    print(f"Fold {fold}")
    fold += 1
    train_data = data.iloc[train_index]
    test_data = data.iloc[test_index]
    # print("test_data:", test_data)

    # Calculamos medias y desviaciones estándar para cada clase en los datos de entrenamiento
    clasesTrainData, _ = getClasesData(train_data)
    medias, desviaciones = getMean_Std(clasesTrainData)
    # print("Medias por clase:", medias)
    # print("Desviaciones estándar por clase:", desviaciones)

    # Clasificamos los datos de prueba
    for index, row in test_data.iterrows():
        posteriors = []
        print("Datos de prueba:", row[1:5].values)
        for i in range(len(clasesData)):
            prior = pW[i]
            # print("Prior:", prior)
            likelihood = getGaussianProbability(row[1:5].values, medias[i], desviaciones[i])
            posterior = math.log(prior * likelihood.prod())
            posteriors.append(posterior)
            print(f"Posterior para clase {i}: {posterior}")
        predicted_class = data['Species'].unique()[np.argmax(posteriors)]
        print(f"Ejemplo {index} - Clase real: {row['Species']} - Clase predicha: {predicted_class}")
