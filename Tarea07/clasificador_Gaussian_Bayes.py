import pandas as pd
import numpy as np
import math
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, balanced_accuracy_score, precision_score
from imblearn.metrics import sensitivity_score, specificity_score

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

def getNombreClases(data, nombreClase):
    nombreClases = data[nombreClase].unique()
    return nombreClases

def getClasesData(data, nombreClase):
    clasesData = []
    nombreClases = getNombreClases(data, nombreClase)
    lenClases = len(nombreClases)
    for clase in nombreClases:
        clasesData.append(data[data[nombreClase] == clase])
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

# 4. Clasificador Gaussian Naive Bayes
def gaussianNaiveBayes(clasesData, pW, muestra, medias, desviaciones, i_Atr, j_Atr):
    epsilon = 1e-10
    numClases = len(clasesData)
    pPosteriores = []
    clasesPredichas = []

    # print("Muestra a clasificar:", muestra)
    # print("Medias:", medias)
    # print("Desviaciones:", desviaciones)
    # print("Numero de clases:", numClases)

    for index, row in muestra.iterrows():
        pPosts = []
        for i in range(numClases):
            probabilidadClase = pW[i]
            likelihood = getGaussianProbability(row[i_Atr:j_Atr].values, medias[i], desviaciones[i])
            likelihood = np.where(likelihood > 0, likelihood, epsilon)
            posterior = math.log(probabilidadClase) + np.sum(np.log(likelihood))
            posterior = float(posterior)
            pPosts.append(posterior)
        #     print(f"Posterior para clase {i}: {posterior}")
        # print("Probabilidades posteriores:", pPosts)
        clasePredicha = np.argmax(pPosts)
        clasePredicha = int(clasePredicha)
        # print("Indice de clase predicha:", clasePredicha)
        
        pPosteriores.append(pPosts)
        clasesPredichas.append(clasePredicha)

        
        
        

    

    

    # posteriors = []
    #     print("Datos de prueba:", row[1:5].values)
    #     for i in range(len(clasesData)):
    #         prior = pW[i]
    #         # print("Prior:", prior)
    #         likelihood = getGaussianProbability(row[1:5].values, medias[i], desviaciones[i])
    #         posterior = math.log(prior) + np.sum(np.log(likelihood))
    #         posterior = float(posterior)
    #         posteriors.append(posterior)
    #         # print(f"Posterior para clase {i}: {posterior}")
    #     print("Probabilidades posteriores:", posteriors)
    #     predicted_class = nombresClases[np.argmax(posteriors)]

    # posteriors = []
        # print("Datos de prueba:", row[1:5].values)
        # for i in range(len(clasesData)):
        #     prior = pW[i]
        #     # print("Prior:", prior)
        #     likelihood = getGaussianProbability(row[1:5].values, medias[i], desviaciones[i])
        #     posterior = math.log(prior * likelihood.prod())
        #     posteriors.append(posterior)
        #     print(f"Posterior para clase {i}: {posterior}")
        # print("Probabilidades posteriores:", posteriors)
    return clasesPredichas, pPosteriores






# Main
data = readData('./Tarea07/Iris.csv')
atributoEtiqueta = 'Species'
nombresClases = getNombreClases(data, atributoEtiqueta)
clasesData, totalClases = getClasesData(data, atributoEtiqueta)


print("Tamaño total de datos:", len(data))
print("Total de clases:", totalClases)
# print("Clases data:", clasesData)
# print(clasesData[2])

# 1. Calculamos la probabilidad a priori de cada clase
pW = getPriorProb(len(data), clasesData)
print("Probabilidades a priori de cada clase:", pW)

# 2. Dividimos los datos en K-Folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_metrics = [] # Para almacenar métricas de cada fold

fold = 1
for train_index, test_index in kf.split(data):
    print("=======================================")
    print(f"Fold {fold}")
    
    train_data = data.iloc[train_index]
    test_data = data.iloc[test_index]
    # print("test_data:", test_data)

    # Calculamos medias y desviaciones estándar para cada clase en los datos de entrenamiento
    clasesTrainData, _ = getClasesData(train_data, atributoEtiqueta)
    medias, desviaciones = getMean_Std(clasesTrainData)
    # print("Medias por clase:", medias)
    # print("Desviaciones estándar por clase:", desviaciones)

    # Clasificamos los datos de prueba
    # for index, row in test_data.iterrows():
    #     predicted_class_index, posteriors = gaussianNaiveBayes(clasesTrainData, pW, row[1:5].values, medias, desviaciones)
    #     predicted_class = nombresClases[predicted_class_index]
    #     # posteriors = []
    #     # print("Datos de prueba:", row[1:5].values)
    #     # for i in range(len(clasesData)):
    #     #     prior = pW[i]
    #     #     # print("Prior:", prior)
    #     #     likelihood = getGaussianProbability(row[1:5].values, medias[i], desviaciones[i])
    #     #     posterior = math.log(prior) + np.sum(np.log(likelihood))
    #     #     posterior = float(posterior)
    #     #     posteriors.append(posterior)
    #     #     # print(f"Posterior para clase {i}: {posterior}")
    #     # print("Probabilidades posteriores:", posteriors)
    #     # predicted_class = nombresClases[np.argmax(posteriors)]
    #     print(f"Ejemplo {index} - Clase real: {row[atributoEtiqueta]} - Clase predicha: {predicted_class}")

    indicesClassPredic, posteriors = gaussianNaiveBayes(clasesTrainData, pW, test_data, medias, desviaciones, 1, 5)
    print("Indices de clases predichas:", indicesClassPredic)
    print("Tamaño de indices de clases predichas:", len(indicesClassPredic))

    clasesPredichas = [nombresClases[index] for index in indicesClassPredic]

     # Mostramos los resultados
    for i, (index, row) in enumerate(test_data.iterrows()):
        print(f"Ejemplo {index} - Clase real: {row[atributoEtiqueta]} - Clase predicha: {clasesPredichas[i]}")

    # Evaluamos el desempeño del clasificador
    y_true = test_data[atributoEtiqueta].values
    y_pred = clasesPredichas
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    sensitivity = sensitivity_score(y_true, y_pred, average='weighted')
    specificity = specificity_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')

    fold_metrics.append({
        'Fold': fold,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': sensitivity,
        'F1-Score': f1,
        'Balanced Accuracy': balanced_acc,
        'Specificity': specificity,
    })

    fold += 1

# Creando un DataFrame para mostrar las métricas de cada fold
df_metrics = pd.DataFrame(fold_metrics)
print(df_metrics)

# mean_row = pd.DataFrame(df_metrics.mean(numeric_only=True)).T
# mean_row.insert(0, 'Fold', 'Promedio')
# df_metrics = pd.concat([df_metrics, mean_row], ignore_index=True)

df_metrics["Fold"] = df_metrics["Fold"].astype(str)

numeric_cols = df_metrics.select_dtypes(include="number").columns
promedios = df_metrics[numeric_cols].mean().to_dict()
df_metrics.loc[len(df_metrics)] = {"Fold": "Promedio", **promedios}

print("\n=== Resultados por fold y promedio final ===")
print(df_metrics)