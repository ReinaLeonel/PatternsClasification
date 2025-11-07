import pandas as pd
import numpy as np
import math
from scipy.io import arff
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, balanced_accuracy_score, precision_score
from imblearn.metrics import sensitivity_score, specificity_score
from tqdm import tqdm
import warnings

def readData(rutaArchivo, formato='csv'):
    if formato == 'csv':
        data = pd.read_csv(rutaArchivo)
    elif formato == 'arff':
        data = arff.loadarff(rutaArchivo)
        data = pd.DataFrame(data[0])
    return data

def getNombreClases(data, nombreClase, formato='csv'):
    nombreClases = data[nombreClase].unique()
    if formato == 'arff':
        nombreClases = [clase.decode('utf-8') if isinstance(clase, bytes) else clase for clase in nombreClases]
    return nombreClases

def getClasesData(data, nombreClase):
    clasesData = []
    nombreClases = getNombreClases(data, nombreClase)
    lenClases = len(nombreClases)
    for clase in nombreClases:
        clasesData.append(data[data[nombreClase] == clase])
    lenPorClase = [len(clase) for clase in clasesData]
    return clasesData, lenClases, lenPorClase

def getPriorProb(lenData, clasesData):
    pW = []
    for clase in clasesData:
        pW.append(len(clase) / lenData)
    return pW

# 2. Calculamos la probabilidad condicional de cada clase (verosimilitud) calculando la media y la desviación estándar de cada atributo
def getMean_Std(clasesData, rango):
    medias = []
    desviaciones = []
    for clase in clasesData:
        medias.append(clase.iloc[:, rango].mean().values)
        desviaciones.append(clase.iloc[:, rango].std().values)
    return medias, desviaciones

# 3. Se multiplican las probabilidades a priori por las verosimilitudes para cada clase y se elige la clase con mayor probabilidad posterior
def getGaussianProbability(x, mean, std):
    elevacion = -((x - mean) ** 2 / (2 * std ** 2))
    elevacion = np.array(elevacion, dtype=np.float64)
    exponent = np.exp(elevacion)
    fdp = (1 / (np.sqrt(2 * np.pi) * std)) * exponent
    return fdp

# 4. Clasificador Gaussian Naive Bayes
def gaussianNaiveBayes(clasesData, pW, muestra, medias, desviaciones, rango):
    epsilon = 1e-10
    numClases = len(clasesData)
    pPosteriores = []
    clasesPredichas = []

    for index, row in muestra.iterrows():
        pPosts = []
        for i in range(numClases):
            probabilidadClase = pW[i]
            likelihood = getGaussianProbability(row[rango].values, medias[i], desviaciones[i])
            likelihood = np.where(likelihood > 0, likelihood, epsilon)
            posterior = math.log(probabilidadClase) + np.sum(np.log(likelihood))
            posterior = float(posterior)
            pPosts.append(posterior)
        clasePredicha = np.argmax(pPosts)
        clasePredicha = int(clasePredicha)
        
        pPosteriores.append(pPosts)
        clasesPredichas.append(clasePredicha)

    return clasesPredichas, pPosteriores

# Main
# data = readData('./Tarea07/Iris.csv', formato='csv')
data = readData('./Tarea07/Support_tickets.csv', formato='csv')
atributoEtiqueta = 'priority'
nombresClases = getNombreClases(data, atributoEtiqueta, formato='csv')
clasesData, totalClases, tamañoPorClase = getClasesData(data, atributoEtiqueta)

# Eliminación de algunas columnas no numéricas si es necesario, menos la de la etiqueta
numericas = data.select_dtypes(include=['number']).columns.tolist()
columna_extra = 'priority'
columnas_finales = numericas + [columna_extra]
data = data[columnas_finales]

print("data: ", data)
print("Número de columnas:", data.shape[1])
print("Columna: ", data.columns[23])

rango = slice(0, 22)
print("rango de atributos:", rango)


print("Tamaño total de datos:", len(data))
print("Total de clases:", totalClases)
print("Nombres de clases:", nombresClases)
print("Tamaño por clase:", tamañoPorClase)

# 1. Calculamos la probabilidad a priori de cada clase
pW = getPriorProb(len(data), clasesData)
print("Probabilidades a priori de cada clase:", pW)

# 2. Dividimos los datos en K-Folds
kf = KFold(n_splits=50000, shuffle=True, random_state=42)

fold_metrics = [] # Para almacenar métricas de cada fold

fold = 1
for train_index, test_index in tqdm(kf.split(data), total=50000, desc="K-Fold Cross Validation"):
    # print("=======================================")
    # print(f"Fold {fold}")
    
    train_data = data.iloc[train_index]
    test_data = data.iloc[test_index]

    # Calculamos medias y desviaciones estándar para cada clase en los datos de entrenamiento
    clasesTrainData, _, _ = getClasesData(train_data, atributoEtiqueta)
    medias, desviaciones = getMean_Std(clasesTrainData, rango)

    indicesClassPredic, posteriors = gaussianNaiveBayes(clasesTrainData, pW, test_data, medias, desviaciones, rango)
    # print("Indices de clases predichas:", indicesClassPredic)
    # print("Tamaño de indices de clases predichas:", len(indicesClassPredic))

    clasesPredichas = [nombresClases[index] for index in indicesClassPredic]

    # Mostramos los resultados
    # for i, (index, row) in enumerate(test_data.iterrows()):
        # print(f"Ejemplo {index} - Clase real: {row[atributoEtiqueta]} - Clase predicha: {clasesPredichas[i]}")

    # Evaluamos el desempeño del clasificador
    y_true = test_data[atributoEtiqueta].values
    y_true = y_true.astype(str)
    # print("y_true:", y_true)
    y_pred = clasesPredichas
    # print("Clases predichas:", np.unique(y_pred))
    # print("Clases reales:", np.unique(y_true))
    with warnings.catch_warnings(): # Ignorar warnings de métricas
        warnings.simplefilter("ignore")
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        sensitivity = sensitivity_score(y_true, y_pred, average='weighted')
        specificity_value = specificity_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')

    fold_metrics.append({
        'Fold': fold,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': sensitivity,
        'F1-Score': f1,
        'Balanced Accuracy': balanced_acc,
        'Specificity': specificity_value,
    })

    fold += 1

# Creando un DataFrame para mostrar las métricas de cada fold
df_metrics = pd.DataFrame(fold_metrics)

df_metrics["Fold"] = df_metrics["Fold"].astype(str)

numeric_cols = df_metrics.select_dtypes(include="number").columns
promedios = df_metrics[numeric_cols].mean().to_dict()
df_metrics.loc[len(df_metrics)] = {"Fold": "Promedio", **promedios}

print("\n=== Resultados por fold y promedio final ===")
print(df_metrics)