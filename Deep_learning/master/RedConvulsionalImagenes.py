# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 09:15:26 2024
 Ejemplo de red neuronal convolucional en Tensorflow
 Librerías: 
     TensorFlow version: 2.17.0
     Keras version: 3.4.1
     Matplotlib: 3.8.4
@author: Luis A. García
"""
# importando librerías necesarias
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
# leyendo los datos y separando en entrenamiento y prueba
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
#%% Definiendo las clases y visualizando las primeras 25 imágenes
class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog',
               'frog', 'horse', 'ship','truck']
# visualizando las 25 primeras imágenes
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
#%% Construcción de la red neuronal convolucional simple
# creando modelo secuencial
model = models.Sequential()
""" Creando capa convolucional 2D para realizar operaciones de convolución en 
    2 dimensiones. Es útil para procesar imágenes.
    Esta capa contiene:
        •	32 filtros o kernels
        •	El tamaño de cada filtro es de 3x3 píxeles 
        •	Función de activación RELU
        •	Esta capa espera recibir elementos de forma (32,32,3), es decir, 
            imágenes de tamaño 32x32 con tres canales, por ejemplo RGB.
""" 
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))) 
""" Agregando capa al modelo que ayuda a reducir el tamaño de las imagenes 
    de la siguiente forma:
        •	MaxPooling2D es la capa que toma la imagen y la hace más pequeña.
        •	(2, 2) Se divide la imagen en pequeños bloques de 2x2 píxeles.
        •	Se encuentra el valor más alto en cada bloque
        •	Se crea la imagen más pequeña usando esos valores más alto
En resumen: La función max-pooling se utiliza para reducir la dimensionalidad de 
los mapas de características (outputs de las capas convolucionales) y para 
destacar las características más importantes.
"""
model.add(layers.MaxPooling2D((2, 2)))
# creando segunda capa convolucional 2D con activación RELU
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# añadiendo capa para reducir dimensionalidad de las imágenes quedándonos con las características más importantes
model.add(layers.MaxPooling2D((2, 2)))
#creando tercera capa convolucional 2D con activación RELU
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#-----------------------------------------------------------------------------#
# Añadiendo capas densas que usan los features para clasificar las imágenes
#-----------------------------------------------------------------------------#
model.add(layers.Flatten()) # convierte la salida a vector unidimensional
# añadiendo capa densa con 64 neuronas y activación RELU
model.add(layers.Dense(64, activation='relu'))
# agregando capa de salida con 10 neuronas
model.add(layers.Dense(10))
#%% Compilación y entrenamiento del modelo
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# entrenando modelo en 10 épocas
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))
#%% Evaluando el modelo para ver el desempeño en el conjunto de prueba
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')

