# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 09:49:06 2024

Desarrollo de una red neuronal Perceptrón multicapa
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
# cargando y preparando el conjunto de datos
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
#definiendo lista con los nombres de las clases
class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
'truck']
# visualizando las primeras 25 imágenes
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([]) 
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
#%% Construcción del modelo Perceptrón multicapa
# Llamando al modelo que es de tipo secuencial
model = models.Sequential([
    # definiendo las dimensiones que tendrán las características de entrada
    layers.Flatten(input_shape=(32, 32, 3)),
    #definiendo capa densa con 512 neuronas y función de activación relu
    layers.Dense(512, activation='relu'),
    # definiendo segunda capa densa con 256 neuronas y función de activación relu
    layers.Dense(256, activation='relu'),
    # definiendo capa de salida con 10 neuronas
    layers.Dense(10)
])
# compilando modelo 
model.compile(optimizer='adam', # Es una combinación de dos metodologías de descenso de gradiente: RMSProp (Root Mean Square Propagation) y Momentum.
              # definiendo función de pérdida para problemas de clasificación multiclase
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              # definiendo métrica de evaluación
              metrics=['accuracy'])
#%% Entrenamiento del modelo y evaluación del modelo
"""
El modelo se entrena con las variables independientes y objetivos (train_images, train_labels),
usando 10 épocas. Al modelo se le evalúa el rendimiento después de cada época con
la línea validation_data=(test_images, test_labels))
"""
history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))
#%% Visualizando la precisión del modelo durante entrenamiento y validación durante las épocas
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
#%% Evaluando rendimiento del modelo en conjunto de prueba
"""
test_loss: Pérdida del modelo en los datos de prueba
test_acc: Precisión del modelo en los datos de prueba
"""
test_loss, test_acc = model.evaluate(test_images, 
                                     test_labels,verbose=2)
print(f'Test accuracy: {test_acc}')