# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 15:13:13 2024

@author: Luis A. García
"""
# importando librerías necesarias
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
# verificando la disponibilidad de la GPU
devices = tf.config.list_physical_devices('GPU')
if devices:
    print(f"GPUs detectadas: {len(devices)}")
    for gpu in devices:
        print(gpu)
else:
    print("No se detectaron GPUs")
# cargando los datos NMIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# redimensionando las imagenes a un canal, en este caso, escala de grises
train_images = train_images.reshape((60000, 28, 28, 1)) 
test_images = test_images.reshape((10000, 28, 28, 1)) 
# normalizando los valores de intensidad en cada pixel de las imágenes
train_images, test_images = train_images / 255.0, test_images / 255.0
#%% Construcción de la red neuronal
model = models.Sequential() 
# agreando primer convolucional con 32 filtros de 3x3, y activación relu
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.Dropout(0.3))  # Apaga el 30% de las neuronas
# reduciendo imagen entregada a la mitad(14x14x32) conservando las características más relevantes
model.add(layers.MaxPooling2D((2, 2)))
# agregando segunda capa convolucional con 64 filtros de 3x3 y activación relu
model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
model.add(layers.Dropout(0.3))  # Apaga el 30% de las neuronas
# reduciendo imagen entregada a la mitad(7x7x64) conservando las características más relevantes
model.add(layers.MaxPooling2D((2, 2)))
# agregando tercer capa convolucional
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Dropout(0.3))  # Apaga el 30% de las neuronas

# aplanando información de salida de la tercer capa
model.add(layers.Flatten())
# aplicando capa densa con 64 neuronas y activación relu
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.3))  # Apaga el 30% de las neuronas
# agregando capa final que clasifica las imagenes, dado esto, utilizamos la función de activación softmax
model.add(layers.Dense(10, activation='softmax')) 

# Configurar el callback de parada temprana
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

#%% Compilación del modelo
"""
compilando modelo con:
    optimizer='adam' 
    loss='sparse_categorical_crossentropy' :  Es la función de pérdida más común en 
        problemas de clasificación multiclase.
    metrics=['accuracy']: mide el porcentaje de veces que el modelo predijo la clase correcta.
"""
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
#%% Entrenamiento del modelo
# Entrenar el modelo con 100 épocas, usando validación y parada temprana
history = model.fit(train_images, train_labels, 
                    epochs=100, 
                    validation_data=(test_images, test_labels), 
                    callbacks=[early_stopping])
#%% Evaluando modelo
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2) 
print(f'\nTest accuracy: {test_acc}') 
# Graficar las curvas de pérdida y precisión
plt.plot(history.history['accuracy'], label='accuracy') 
plt.plot(history.history['val_accuracy'], label = 'val_accuracy') 
plt.xlabel('Epoch') 
plt.ylabel('Accuracy') 
plt.ylim([0, 1]) 
plt.legend(loc='lower right') 
plt.show() 