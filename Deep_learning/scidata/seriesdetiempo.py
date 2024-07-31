# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:11:36 2024
 Series temporales:
     Vamos a usar un dataset que contiene el número de pasajeros de avión internacionales por mes.
     En concreto vamos a intentar predecir el número de pasajeros de líneas aereas los últimos meses del dataset.
@author: Luis A. García 
"""
# importando librerías necesarias
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
# accediendo a la ruta donde están los datos
os.chdir('D:\\3. Cursos\\11. Deep learning\\Deep_Learning_23-main\\tablas')
# leyendo archivo
pasajeros = pd.read_csv("international-airline-passengers.csv",sep=";")
#nombrando columnas
pasajeros = pasajeros.rename(columns={'Month':'mes','Passengers':'pasajeros'})
"""
La variable objetivo de este modelo será predecir el volumen de pasajeros del mes
siguiente. Podemos usar la función de pandas shift para mover la columna del 
número de pasajeros una posición hacia arriba.
"""
#%% jet -1 series de tiempo
"""
# Crear una nueva columna "pasajeros_1" con los valores de la columna "pasajeros"
# desplazados una posición hacia arriba (equivalente a un desfase temporal de -1).
# El último valor será NaN debido al desplazamiento.
"""
pasajeros["pasajeros_1"] = pasajeros["pasajeros"].shift(-1)
#Al hacer esto la última fila no tiene un valor para predecir, la eliminamos.
pasajeros = pasajeros.drop(143)
# seleccionando la cnatidad de pasajeros y convirtiendo a array float
pasajeros_x = pasajeros["pasajeros"].astype(float).values
pasajeros_y = pasajeros["pasajeros_1"].astype(float).values
#%% Separación en entrenamiento y prueba conservando la tendencia
""" En series de tiempo los datos deben separarse en entrenamiento y prueba
    conservado la tendencia de los datos 
"""
# calculando cantidad de filas
n_periodos = len(pasajeros)
# definiendo un 20 % para prueba
pct_test = 0.2
# definiendo los datos de entrenamiento
n_train = int(n_periodos * (1-pct_test))
n_train
#%% Estandarizando los datos
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(pasajeros_x.reshape(-1, 1))
pasajeros_x_std = scaler.transform(pasajeros_x.reshape(-1, 1))
pasajeros_y_std = scaler.transform(pasajeros_y.reshape(-1, 1))
#%% Separando en entrenamiento y prueba
x_train = pasajeros_x_std[:n_train]
x_test = pasajeros_x_std[n_train:]

y_train = pasajeros_y_std[:n_train]
y_test = pasajeros_y_std[n_train:]
# ajustando las tablas
x_train = x_train.reshape(-1,1,1)
x_test = x_test.reshape(-1,1,1)
#%% Implementado el modelo
from keras import Sequential
from keras.layers import Dense,GRU
# Inicializar el modelo secuencial
modelo_lstm = Sequential()
# Añadir una capa GRU con 32 unidades, devolviendo la secuencia completa, y especificar la forma de entrada
modelo_lstm.add(GRU(32, return_sequences=True, input_shape=(1, 1)))
# Añadir una segunda capa GRU con 32 unidades, devolviendo la secuencia completa
modelo_lstm.add(GRU(32, return_sequences=True))
# Añadir una tercera capa GRU con 32 unidades, devolviendo solo la última salida
modelo_lstm.add(GRU(32))
# Añadir una capa densa con una unidad de salida
modelo_lstm.add(Dense(1))
# Mostrar el resumen de la arquitectura del modelo
modelo_lstm.summary()
#%% Tenemos un problema de predicción de valor
# por lo cuál, la métrica usada será el error cuadrático medio
modelo_lstm.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
modelo_lstm.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1);
#%% Evaluando el modelo en el conjunto de entrenamiento
from sklearn.metrics import mean_squared_error
"""
Aplica la transformación inversa para deshacer el escalado aplicado previamente 
a los datos de entrenamiento, devolviendo las predicciones a su escala original.
"""
train_pred = scaler.inverse_transform(modelo_lstm.predict(x_train))
"""
Aplica la transformación inversa al conjunto de etiquetas de entrenamiento y_train
para devolverlas a su escala original.
"""
y_train_original = scaler.inverse_transform(y_train)
# calculando error cuadrático medio y la raíz cuadrada del MSE
error_train = np.sqrt(mean_squared_error(y_train_original, train_pred))
error_train
#%% Evaluando el modelo en el conjunto de prueba
from sklearn.metrics import mean_squared_error

test_pred = scaler.inverse_transform(modelo_lstm.predict(x_test))
y_test_original = scaler.inverse_transform(y_test)
error_test = np.sqrt(mean_squared_error(y_test_original, test_pred))

error_test
#%% procedemos a graficar 
# creando arreglo de ceros
test_pred_plot = np.zeros(pasajeros_y.shape)
# llenando con las predicciones
test_pred_plot[-test_pred.shape[0]:] = test_pred[:,0]
# reemplazando con nan los valores que no se llaron 
test_pred_plot[:-test_pred.shape[0]] = np.nan
# graficando 
plt.plot(pasajeros_y)
plt.plot(train_pred, label="predicción train")
plt.plot(test_pred_plot, label="predicción test")
plt.title("Número de pasajeros internacionales")
plt.legend();