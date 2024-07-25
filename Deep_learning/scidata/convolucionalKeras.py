# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 17:04:29 2024
 Redes neuronales de convolución

@author: Luis A. García 
"""
# Importando librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
"""
La convolución es una operación entre matrices donde una matriz pequeña 
(kernel o filtro) se aplica de forma iterativa a los distintos trozos de una 
matriz más grande (una imagen).

En primer lugar vamos a ver de forma práctica el ejemplo de convolución visto 
en la teoría. Para ello creamos una imagen con dos troz verticales, 
uno blanco y otro negro
"""
#%%  detección de bordes verticales
from scipy import signal
import scipy.datasets
# cargando imagen de prueba
img_original  = scipy.datasets.ascent()
# graficando
plt.imshow(img_original, cmap='gray')
plt.title("Imagen Original");
# definiendo filtro vertical
filtro_sobel_vertical = np.array([[1, 0, -1],
                                  [1, 0, -1],
                                  [1, 0,-1]])
# aplicando filtro de detección de bordes verticales
output_convolucion = signal.convolve2d(img_original, filtro_sobel_vertical)
plt.imshow(np.absolute(output_convolucion), cmap='gray')
plt.title("Output de la convolución con filtro vertical");
#%% Detección de bordes horizontales
# defiendo filtro horizontal
filtro_sobel_horizontal = np.array([[-1,-1,-1],
                                    [0, 0, 0],
                                    [1, 1,1]])
# aplicando filtro para detectar bordes horizontales
output_convolucion = signal.convolve2d(img_original, filtro_sobel_horizontal)
plt.imshow(np.absolute(output_convolucion), cmap='gray')
plt.title("Output de la convolución con filtro horizontal");
#%% 
""" En esta sección vamos a analizar el sed de datos Fashion MNIST que es un 
Datased de ropa (hoy en día se considera un datased demasiado fácil).
En concreto el dataset Fashion MNIST consiste en 60,000 fotos 
(28x28 pixeles como el dataset de dígitos) de prendas de ropa y su objetivo es 
clasificar las imágenes respecto al tipo de ropa mostrado (10 tipos distintos de artículos)
"""
from keras.utils import to_categorical
from keras.datasets import fashion_mnist
# cargando los datos y dividiendo en entrenamiento y pueba
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# convirtiendo la variable objetivo a hot-encoder 
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)
# visualizando algunas imágenes
def dibujar_imagen(i):
    plt.imshow(x_train[i], cmap="gray")
    plt.title("Clase de prenda: {}".format(y_train[i]))
    
dibujar_imagen(10)
#%% Preparando las imágenes para ser usadas en  red neuronal convolucional
#difiendo tañano de la imagen
img_rows, img_cols = 28, 28
"""
Preparando las imagenes para ser utilizadas en el modelo de aprendizaje
x_train.shape[0] representa el número de imágenes en el conjunto de entrenamiento.
img_rows y img_cols definen la altura y el ancho de cada imagen
1 indica que las imágenes son en escala de grises (un solo canal de color).
"""
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1) # caracteristicas de entrada a la red neuronal
#%% construyendo red neuronal convolucional
from keras.layers import Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense, Dropout 
# definendo tamaño de paso
batch_size = 256
# definiendo número de clases
num_classes = 10
# definiendo épocas
epochs = 50
# construyendo red de tipo secuencial
modelo_cnn = Sequential()
# añadiendo capa de convolución 2D con 32 matrices de 3x3
modelo_cnn.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
"""añadiendo capa maxpooling2d que reduce el tamaño de las imágenes de salida de
    la red convolucional manteniendo las características más importantes
"""
modelo_cnn.add(MaxPooling2D(pool_size=(2, 2)))
# implementado regularización Dropout donde le decimos a la red que apague el 25% de las neuronas en cada iteración
modelo_cnn.add(Dropout(0.25))
# aplanando resultados (imagenes) para ser usados en capas densas
modelo_cnn.add(Flatten())
# implementando capa densa de 32 neuronas
modelo_cnn.add(Dense(32, activation='relu'))
# impementado regularización Dropout que apaga el 50% de las neuronas
modelo_cnn.add(Dropout(0.5))
modelo_cnn.add(Dense(num_classes, activation='softmax'))
modelo_cnn.summary()
#%% Compilando red neuronal
"""
optimizer="adam" # adam es una versión mejorada del desenso de gradiente
la métrica será la exactitud
"""
modelo_cnn.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
"""# ajustando modelo en donde se utilizan 1000 muestras 
Si tienes un conjunto de datos de 10,000 imágenes y batch_size=1000, 
cada época del entrenamiento consistirá en 10 iteraciones (10,000 / 1,000). 
En cada iteración, el modelo ajusta sus parámetros basándose en las 1,000 
imágenes procesadas en esa iteración. Luego, pasa al siguiente lote de 1,000 
imágenes, y así sucesivamente, hasta que haya procesado todos los datos una 
vez por época.
"""
modelo_cnn.fit(x_train, y_train_one_hot, epochs=50, batch_size=1000, verbose=1);
# evaluando modelo
modelo_cnn.evaluate(x_test, y_test_one_hot, verbose=0)
#%% Generalmente se concatenan multiples capas en una red CNN
from keras.layers import BatchNormalization


batch_size = 128
num_classes = 10
epochs = 100
filter_pixel=3
noise = 1
droprate=0.25

model = Sequential()
"""
creando capa de 64 matrices de tamaño 3x3 y añadimos Padding ()
    Padding consiste en añadir ceros alrededor de la imagen para que las
    dimensiones de la salida no se alteren.
"""
model.add(Conv2D(64, kernel_size=(filter_pixel, filter_pixel), padding="same",
                 activation='relu',
                 input_shape=input_shape)) 
# añadimos normalización a las salidas 
model.add(BatchNormalization())
# añadiendo regularización Dropout que apaga neuronas aleatoriamente
model.add(Dropout(droprate))
# añadiendo segunda capa convolucional
model.add(Conv2D(64, kernel_size=(filter_pixel, filter_pixel), activation='relu',padding="same"))#1
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(droprate))
# añadiendo tercera capa convolucional
model.add(Conv2D(64, kernel_size=(filter_pixel, filter_pixel), activation='relu',padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(droprate))
# aplanando los resultados
model.add(Flatten())
#añadiendo capas densas
model.add(Dense(500,use_bias=False, activation="relu")) 
model.add(BatchNormalization())
model.add(Dropout(droprate))      

model.add(Dense(10, activation="softmax"))

model.summary()
#%%
from keras.callbacks import EarlyStopping
"""
EarlyStopping: Un callback que monitorea una métrica y detiene el entrenamiento si dicha métrica deja de mejorar.
monitor='val_accuracy': Monitorea la precisión de validación (val_accuracy).
patience=10: El entrenamiento se detendrá si val_accuracy no mejora en 10 épocas consecutivas.
mode='max': Indica que se está buscando maximizar la métrica monitoreada (val_accuracy).
verbose=1: Muestra mensajes cuando el entrenamiento se detiene temprano.
"""
callbacks = [
    EarlyStopping(
        monitor='val_accuracy', 
        patience=10,
        mode='max',
        verbose=1)
]

# compilando el modelo
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])
# Entrenamiento del modelo
"""
x_train, y_train_one_hot: Los datos de entrada y las etiquetas de entrenamiento en formato one-hot.
batch_size=batch_size: El tamaño del lote a utilizar en cada iteración del entrenamiento.
epochs=1: Número de épocas para entrenar el modelo. Aunque está configurado a 1 en este ejemplo, generalmente se entrena por más épocas.
verbose=1: Muestra el progreso del entrenamiento en la consola.
validation_data=(x_test, y_test_one_hot): Conjunto de datos de validación para evaluar el modelo después de cada época.
shuffle=True: Baraja los datos de entrenamiento antes de cada época para ayudar a que el modelo generalice mejor.
callbacks=callbacks: Pasa la lista de callbacks al método fit, en este caso, solo EarlyStopping.
"""
history = model.fit(x_train, y_train_one_hot,
          batch_size=batch_size,
          epochs=5,
          verbose=1,
          validation_data=(x_test, y_test_one_hot), 
          shuffle=True,callbacks=callbacks)
#%% evaluando modelo
model.evaluate(x_test, y_test_one_hot, verbose=1)