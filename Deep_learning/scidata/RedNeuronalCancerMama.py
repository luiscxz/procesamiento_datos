# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 08:34:48 2024
Ejemplo de como trabajar en keras para crear modelos de deep learning
@author: Luis A. García
"""
# importando librerías necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 10)
# procedemos a cargar los datos
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
# cargando tabla de cancer de mama
mi_bd = load_breast_cancer()
# seleccionando variables indenpendientes y objetivo
X, y = mi_bd.data, mi_bd.target
# seleccionando solo las primeras 4 columnas
X = mi_bd.data[:,:4]
# procedemos a estandarizar los datos
X_std = StandardScaler().fit_transform(X)
y = y.reshape(569,1)
"""
Keras por si mismo no se encarga de hacer todas las operaciones de bajo nivel 
(operaciones matriciales), sino que soporta varios backends 
(el motor que hará el entrenamiento), podemos elegir el que queremos activando 
la variable de entorno KERAS_BACKEND.

Keras soporta los siguientes backends:

theano: Librería de deep learning original de python para deep learning. Hoy en
dia raramente se usa por si sola.
tensorflow: Librería de deep learning desarrollada por google.
CNTK Librería de deep learning desarrollada por Microsoft
[Pytorch]
"""
#%% Activando backend tendorflow
import os
os.environ["KERAS_BACKEND"]="tensorflow"
#%% Red neuronal
from keras.models import Sequential
from keras.layers import Dense
#llamando al modelo que es de tipo secuencial
modelo = Sequential()
"""
# nota: Nunca se programa la capa de entrada
# procedo a añadir una capa densa de 5 neuronas, función de activacion relu  y 
# le decimos que la capa espera 4 características
"""
modelo.add(Dense(units=5, activation='relu', input_shape=(4,)))
# Añadiendo capa densa de salida con 1 neurona y función de activación Sigmoid
modelo.add(Dense(units=1, activation='sigmoid')) 
# otra forma de hacer lo mismo es:
"""
modelo = Sequential([
    Dense(units=5, activation='relu', input_dim=4),
    Dense(units=1, activation='sigmoid')
])
"""
#%% Procedemos a compilar el modelo y definir la función de pérdidas que medirá el error propagado
# como estamos resolviendo un problema de clasificación binaria nos conviene utilizar la 
# función de pérdida logarítmica. Tambien le puedo específicar si quiero que me
# tome otras médidas, por ejemplo, la exactitud (accuracy). Adicionalmente le puedo
# decir que utilice el gradiente descendente SGD

modelo.compile(loss="binary_crossentropy", # función de pérdida
              optimizer="sgd", # gradiente descendente
              metrics=["accuracy"]) # presición 
#%%Si queremos modificar los parámetros del optimizador tenemos que crear el objeto optimizador.
#Keras soporta SGD pero tambien muchos otros.
from tensorflow.keras.optimizers import SGD
# defiendo gradiente descendente con radio de aprendizaje de 0.01
sgd = SGD(learning_rate=0.01)
# con esta modificación, vuelvo a compilar el modleo
modelo.compile(loss="binary_crossentropy", # función de pérdida
              optimizer=sgd, # gradiente descendente
              metrics=["accuracy"]) # presición 

# procedemos a ver una descripción general del modelo
modelo.summary()
#%% Procedemos a ajustar el modelo con las variables independientes y objetivos usando 100 épocas
historial = modelo.fit(X_std, y , epochs=100,verbose=0)
#solicitando que me muestre como se comportó el modelo en las 100 épocas
historial.history.keys()
# convirtiendo historia a pandas
historia_Epoca= pd.DataFrame(historial.history)
# graficando exactitud vs época
plt.plot(historial.history["accuracy"])
plt.title("Exactitud vs épocas de entrenamiento");
#%% Procedemos a realizar predicciones y evaluar el rendimiento del modelo
modelo.predict(X_std)[:5] # en este caso se usó la misma tabla, pero no es lo que se debe hacer
# evaluando rendimiento del modelo
scores = pd.DataFrame([modelo.evaluate(X_std, y)],
                      columns =modelo.metrics_names)
# procedemos a obtener los pesos
modelo.get_weights()