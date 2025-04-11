#python -m venv sklearn-env
#sklearn-env\Scripts\activate  # activate
#pip install -U scikit-learn

#pip install numpy
#pip install pandaspython -m pip install -U pip
#python -m pip install -U matplotlib

#pip install seaborn




def determinar_mejor_kernel(X_train, X_test, y_train, y_test):
    """
    Determina el mejor kernel basado en las métricas de desempeño.
    """
    # Definir kernels para SVM
    kernels = ['linear', 'poly', 'rbf']
    degrees = [2, 3]  # Grados para el kernel polinomial
    resultados = []

    # Entrenar modelos con diferentes kernels
    for kernel in kernels:
        if kernel == 'poly':
            for degree in degrees:
                model = svm.SVC(kernel=kernel, degree=degree)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                resultados.append((f'{kernel} (grado {degree})', acc, model))
        else:
            model = svm.SVC(kernel=kernel)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            resultados.append((kernel, acc, model))

    # Encontrar el kernel con la mayor Accuracy
    mejor_kernel = max(resultados, key=lambda x: x[1])  # Seleccionar por Accuracy
    print(f"El mejor kernel es: {mejor_kernel[0]} con Accuracy: {mejor_kernel[1]:.4f}")
    return mejor_kernel[2]  # Retornar el modelo correspondiente al mejor kernel


def clasificar_con_mejor_kernel():
    """
    Clasifica nuevas muestras usando el mejor kernel encontrado.
    """
    # Leer datos (ajusta esto según tu archivo CSV u origen de datos)
    df = pd.read_csv('dd.csv')
    X = df.iloc[:, :-1].values  # Características
    y = df.iloc[:, -1].values  # Variable objetivo

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Determinar el mejor kernel
    mejor_modelo = determinar_mejor_kernel(X_train, X_test, y_train, y_test)

    # Solicitar datos de entrada para clasificar una nueva muestra
    print("\nIngrese los valores de las características para clasificar una nueva muestra:")
    nueva_muestra = []
    for i in range(X.shape[1]):  # Iterar sobre el número de características
        valor = float(input(f"Característica {i + 1}: "))
        nueva_muestra.append(valor)

    # Clasificar la nueva muestra
    clase_predicha = mejor_modelo.predict([nueva_muestra])
    print(f"\nLa clase predicha para la nueva muestra es: {clase_predicha[0]}")

# 


# Importar bibliotecas necesarias
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.cluster.hierarchy import fcluster
import sys

# Función para uniformar los nombres de los clusters con las clases de y
def uniformar_clusters_con_clases(y, clusters):
    """
    Uniforma los nombres de los clusters para que coincidan con las clases de y.
    
    Parámetros:
    - y: array-like, las etiquetas originales de las clases.
    - clusters: array-like, las etiquetas de los clusters asignados.
    
    Retorna:
    - clusters_uniformados: array-like, los clusters renombrados para coincidir con las clases de y.
    """
    unique_classes = np.unique(y)  # Clases únicas en y
    unique_clusters = np.unique(clusters)  # Clusters únicos
    
    # Crear un diccionario para mapear clusters a clases
    cluster_to_class = {}
    
    for cluster in unique_clusters:
        # Encontrar la clase más frecuente dentro del cluster
        indices_cluster = np.where(clusters == cluster)[0]  # Índices de este cluster
        clases_en_cluster = y[indices_cluster]  # Etiquetas de y en este cluster
        clase_mas_frecuente = np.argmax(np.bincount(clases_en_cluster))  # Clase más común
        
        # Asignar el cluster a esta clase
        cluster_to_class[cluster] = clase_mas_frecuente
    
    print("Mapeo de clusters a clases:", cluster_to_class)
    
    # Renombrar los clusters según el mapeo
    clusters_uniformados = np.array([cluster_to_class[cluster] for cluster in clusters])
    
    return clusters_uniformados


# Leer el archivo CSV
df = pd.read_csv('dd.csv')

# Verificar si hay valores faltantes
print("Valores faltantes en el DataFrame:")
print(df.isnull().sum())

# Eliminar filas con valores faltantes
df = df.dropna()

# Extraer las características (todas las columnas excepto la última)
X = df.iloc[:, :-1].values  # Todas las columnas menos la última
y = df.iloc[:, -1].values   # La última columna es la variable objetivo (y)

# Convertir y a enteros si es necesario
if not np.issubdtype(y.dtype, np.integer):
    print("Convirtiendo y a tipo entero...")
    y = y.astype(int)

unique_classes = np.unique(y)
print("Clases únicas en y:", unique_classes)
class_counts = pd.Series(y).value_counts()
print("Distribución de clases:")
print(class_counts)

# Verificar si el número de clases únicas no es 2
if len(unique_classes) != 2:
    print("Error: El número de clases únicas no es 2. Última columna debe ser de dos valores (y).")
    sys.exit()

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizar los datos entre 0 y 1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------
# Materiales y Métodos
# -------------------------

# Clustering jerárquico aglomerativo (HACA)
# Calcular la matriz de enlace utilizando el método de Ward
linked = linkage(X, method='ward')

# Crear el dendrograma para visualizar los clusters
plt.figure(figsize=(12, 8))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrograma de Clustering Jerárquico Aglomerativo')
plt.xlabel('Arqueros')
plt.ylabel('Distancia')
plt.axhline(y=30, color='r', linestyle='--')  # Línea para identificar los clusters más significativos
plt.show()

# Asignar clusters basados en una distancia umbral
clusters = fcluster(linked, t=4, criterion='distance')  # t=4 es un umbral de distancia sugerido

# Uniformar los nombres de los clusters con las clases de y
clusters_uniformados = uniformar_clusters_con_clases(y, clusters)
df['Cluster'] = clusters_uniformados

# Mostrar los clusters asignados
print("\nAsignación de Clusters:")
cluster_counts = df['Cluster'].value_counts()
print(cluster_counts)

# Identificar los índices de los arqueros en cada cluster
cluster_indices = {}
for cluster_id in sorted(df['Cluster'].unique()):
    indices = df[df['Cluster'] == cluster_id].index.tolist()
    cluster_indices[cluster_id] = indices

# Mostrar los índices de los arqueros en cada cluster
print("\nÍndices de los arqueros en cada cluster:")
for cluster_id, indices in cluster_indices.items():
    print(f"Cluster {cluster_id}: {indices}")

# *************************************************
# Boxplots para comparar variables entre HPA y LPA
# *************************************************
df['Cluster'] = y  # Añadir la columna de clusters al DataFrame
df_melt = pd.melt(df, id_vars=['Cluster'], value_vars=df.columns[:-2],
                  var_name='Variable', value_name='Valor')
plt.figure(figsize=(15, 10))
sns.boxplot(x='Variable', y='Valor', hue='Cluster', data=df_melt)
plt.title('Comparación de Variables entre HPA y LPA')
plt.xticks(rotation=45)
plt.show()

# -------------------------
# Entrenamiento y Prueba del Modelo SVM
# -------------------------

# Definir kernels para SVM
kernels = ['linear', 'poly', 'rbf']
degrees = [2, 3]  # Grados para el kernel polinomial (cuadrático y cúbico)
models = []

# Entrenar modelos SVM con diferentes kernels
for kernel in kernels:
    if kernel == 'poly':
        for degree in degrees:
            model = svm.SVC(kernel=kernel, degree=degree)
            model.fit(X_train, y_train)
            models.append((f'{kernel} (grado {degree})', model))
    else:
        model = svm.SVC(kernel=kernel)
        model.fit(X_train, y_train)
        models.append((kernel, model))

# Evaluar modelos
results = []
for name, model in models:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    results.append({
        'Kernel': name,
        'ACC (%)': acc * 100,
        'SENS (%)': report['1']['recall'] * 100,
        'SPEC (%)': report['0']['recall'] * 100,
        'PREC (%)': report['1']['precision'] * 100,
        'ERR (%)': (1 - acc) * 100,
        'MCC': np.corrcoef(y_test, y_pred)[0, 1]
    })

# Crear tabla de resultados (Tabla 2)
results_df = pd.DataFrame(results)
print(results_df)

# -------------------------
# Visualización de Matrices de Confusión
# -------------------------
# Definir los encabezados y el contenido de la tabla
header = "Confusion Matrix."
predicted_class_header = "Predicted class"
actual_class_header = "Actual Class"

# Contenido de la tabla
table_content = """
| TN (True Negative)  |FP (False Positive) |
|---------------------|--------------------|
| FN (False Negative) |TP (True Positive)  |
|---------------------|--------------------|
"""
print("Accuracy (ACC)           ACC = (TP + TN) / (TP + TN + FP + FN)")
print("Sensitivity (SENS).      SENS = TP / (TP + FN)")
print("Specificity (SPEC).      SPEC = TN / (TN + FP)")
print("Precision (PREC).        PREC = TP / (TP + FP)")
print("Error Rate (ERR).        ERR = (FP + FN) / (TP + TN + FP + FN)")
print("Coeficiente de Correlación Matemática (MCC).")
print("  MCC = ((TP * TN) - (FP * FN)) / sqrt((TP + FN) * (TP + FP) * (TN + FN) * (TN + FP))")


# Imprimir la tabla
print(header.center(50, '-'))  # Centrar el título
print(predicted_class_header.center(50))  # Centrar el encabezado de la clase predicha
print(table_content)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()


for i, (name, model) in enumerate(models):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues', cbar=False)
    axes[i].set_title(name)
    axes[i].set_xlabel('Predicho')
    axes[i].set_ylabel('Real')

plt.tight_layout()
plt.show()

clasificar_con_mejor_kernel()