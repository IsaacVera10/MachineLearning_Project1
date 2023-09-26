# Proyecto1_Machine_Learning
Clasificación: Regresión Lineal

## Tabla de contenido
- [**Proyecto1\_Machine\_Learning**](#proyecto1_machine_learning)
    - [**Tabla de contenido**](#tabla-de-contenido)
    - [**Instalación**](#instalación)
    - [**Ejecución**](#ejecución)
    - [**Estructura de Carpetas**](#estructura-de-carpetas)
    - [**Resultados**](#resultados)
    - [**Informe Detallado**](#informe-detallado)
        - [**Introducción**](#introducción)
        - [**Explicación de los Modelos y Extracción de Características**](#explicación-de-los-modelos-y-extracción-de-características)
            - [**Modelos utilizados**](#modelos-utilizados)
                - [**Regresión Logística (LR)**](#regresión-logística-(lr))
                - [**Support Vector Machine (SVM)**](#support-vector-machine-(svm))
                - [**K Nearest Neighbors (KNN)**](#k-nearest-neighbors-(knn))
                - [**Árboles de Decisión**](#árboles-de-decisión)
            - [**Extracción de Características**](#extracción-de-características)
                - [**Redimensionamiento de Imágenes**](#redimensionamiento-de-imágenes)
                - [**Extracción de Características con DWT**](#extracción-de-características-con-dwt)

## Instalación
* Creación de un entorno virtual:
    ```
    python -m venv env
    ```
* Activación del entorno virtual 
    * En Windows:
        ```
        env\Scripts\activate
        ```
    * En Linux:
        ```
        source env/bin/activate
        ```
* Instalación de las librerías necesarias:
    ```
    pip install -r requirements.txt
    ```
## Ejecución
   * Uso de Markdown: Solo debemos ejecutar todas las celdas o cada celda, tendremos que elegir el kernel de Python 3.


## Estructura de Carpetas
images/: Contiene 832 imágenes de mariposas que se utilizarán para el entrenamiento y la evaluación del modelo.

## Resultados
Después de ejecutar el Notebook, obtendrás los siguientes resultados:

- Precisión de Regresión Logística.
- Precisión de SVM.
- Precisión de KNN.
- Precisión de Árboles de decisión.
- Precisión media y desviación estándar de la precisión para cada modelo mediante validación cruzada.

## Informe Detallado

### Introducción

Este proyecto tiene como objetivo clasificar diferentes especies de mariposas utilizando técnicas de procesamiento de imágenes y aprendizaje automático. Se proporciona un archivo Jupyter Notebook (main.ipynb) que contiene el código necesario para realizar la clasificación. Las imágenes de mariposas se encuentran en la carpeta `images/`.

### Explicación de los Modelos y Extracción de Características

### Modelos utilizados
##### **Regresión Logística (LR)** 
La regresión logística es un modelo de clasificación lineal que se utiliza comúnmente para problemas de clasificación binaria y multiclase. En nuestro proyecto, utilizamos el siguiente código para entrenar y evaluar un modelo de Regresión Logística:

```python
# Entrenar y evaluar Regresión Logística
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
```

##### **Support Vector Machine (SVM)** 
Las máquinas de vectores de soporte son modelos de clasificación que buscan encontrar un hiperplano que maximice el margen entre las clases en el espacio de características. Utilizamos SVM con el siguiente código:

```python
# Entrenar y evaluar SVM
svm = SVC()
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
```

##### **K Nearest Neighbors (KNN)** 
KNN es un modelo de clasificación basado en la similitud. Clasifica un punto de datos en función de la mayoría de sus k vecinos más cercanos en el espacio de características. Utilizamos KNN de la siguiente manera:

```python
# Entrenar y evaluar KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)
```

##### **Árboles de Decisión** 
Los árboles de decisión son modelos de clasificación que dividen recursivamente el espacio de características en regiones más pequeñas y homogéneas. Utilizamos árboles de decisión de la siguiente manera:

```python
# Entrenar y evaluar Árboles de decisión
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)
tree_accuracy = accuracy_score(y_test, tree_pred)
```

### Extracción de Características

##### **Redimensionamiento de Imágenes** 
Para garantizar que todas las imágenes tengan el mismo tamaño y facilitar el procesamiento, redimensionamos todas las imágenes a un tamaño común de 1024x768 píxeles utilizando la función resize_image:

```python
def resize_image(image, target_size):
    return resize(image, target_size, mode='constant', anti_aliasing=True)
```

##### **Extracción de Características con DWT** 
La extracción de características se realiza en la función Get_Feacture. En esta función, aplicamos la transformada discreta de ondícula 2D (DWT) utilizando la biblioteca pywavelets. Luego, aplanamos la imagen resultante para obtener un vector de características.

```python
def Get_Feacture(picture, cortes, target_size=(1024, 768)):
    # Resize the image to a common size (e.g., 64x64)
    picture = resize_image(picture, target_size)
    
    LL = picture
    for i in range(cortes):
        LL, (LH, HL, HH) = pywt.dwt2(LL, 'haar')
    return LL.flatten()
```

### Experimentos:
### Pruebas:
### Conclusiones:

