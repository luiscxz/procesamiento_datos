# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:05:39 2024
* Regularización en keras
* Usaremos la base de datos mnist: Esta contiene una colección de números entre
    0 y 9 (son imagenes)
@author: Luis A. García
"""
# Importando librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
"""
Nota: En deep learning no se suele hacer validación cruzada por el costo 
computacional (a menos que el tamaño del datased lo permita). En lugar de 
eso, se hacen simples separaciones entre datos de entrenamiento y validación
"""
# Leyendo los datos  y separando en entrenamiento y prueba
(x_train, y_train), (x_test, y_test)= mnist.load_data()
# creando función que grafica la imagen del número
def dibujar_numero(i):
    plt.imshow(x_train[i], cmap="gray")
    plt.title("Número {}".format(y_train[i]))
# graficando 
dibujar_numero(20)
#%% Definiendo backend TensorFlow
import os
os.environ["KERAS_BACKEND"] = "tensorflow" #tensorflow
# con sultando dimensiones de las variables
x_train.shape
# consultando las clases
np.unique(y_train)
"""
Convirtiendo con junto de imagenes de 28x28 a vector de 784 elementos.
Nota: x_train.shape[0] indica la cantidad de imagenes que hay en la variables
esto da como resultado que cada fila represente una imagen.
"""
x_train_plano = x_train.reshape(x_train.shape[0],28*28)
x_test_plano = x_test.reshape(x_test.shape[0],28*28)
#%% Codificando las variables objetivo a variables dummy.
"""
Nota: Eso se hace ya que a los modelos les es más fácil trabar con variables dummy
"""
from keras.utils import to_categorical
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)
#%% Creando modelo de tipo secuecial y capas densas
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
modelo = Sequential()
# creando capa densa con 50 neurona, activación relu y que espera un vector con 784 características
modelo.add(Dense(50, activation="relu", input_shape=(784,)))
# agregando segunda capa densa con 254 neuronas y activacion relu
modelo.add(Dense(250, activation="relu"))
#Agregar una capa de salida a la red neuronal que contiene la misma cantidad de neuronas que el número de clases.
# se usa activation="softmax" ya que es un problema de multiclase
modelo.add(Dense(np.unique(y_train).shape[0], activation="softmax"))
#procedemos a compliar con modelo usando, y dado que es multiclase se usa entropia categórica
modelo.compile(optimizer="sgd", loss="categorical_crossentropy",
               metrics=["accuracy"])
# obtiendo resumen del modelo
modelo.summary()
#%% Entrenamiento del modelo
modelo.fit(x_train_plano, y_train_one_hot, epochs=300, batch_size=1000, verbose=0);
# procedmeos a evaluar el rendimiento del modelo durante el entrenamiento
evaluacion_train = modelo.evaluate(x_train_plano, y_train_one_hot)
evaluacion_train
#%% Regularizaciones con keras 
resultados ={}
from keras import regularizers
# creando modelo secuencial
modelo_l2 = Sequential()
# creando capa con 50 neuronas, activación RELU y que espera un vector de 784 caracteristicas
modelo_l2.add(Dense(50, activation="relu", input_shape=(784,)))
#creabdi segunda capa densa con 50 neuronas, activación RELU y regularización l2 con lambda de 0.01
modelo_l2.add(Dense(250, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
#creando capa de salida con activación multiclase
modelo_l2.add(Dense(np.unique(y_train).shape[0], activation="softmax"))
# compilando el modelo
modelo_l2.compile(optimizer="sgd", loss="categorical_crossentropy",
               metrics=["accuracy"])

modelo_l2.summary()
# ajustando modelo
modelo_l2.fit(x_train_plano, y_train_one_hot, verbose=0, epochs=50, batch_size=1000);
# verificando la como le fue en exactitud al modelo en zona de entrenamiento
acc_train = modelo_l2.evaluate(x_train_plano, y_train_one_hot)[1]
acc_train
# verificando la como le fue en exactitud al modelo en zona de test
acc_test = modelo.evaluate(x_test_plano, y_test_one_hot)[1]
acc_test
# guardando resultados en el diccionario
resultados['regularización l2']=[acc_train, acc_test]
#%% Regularización L1 con keras
modelo_l1 = Sequential()
modelo_l1.add(Dense(50, activation="relu", input_shape=(784,)))
modelo_l1.add(Dense(250, activation="relu", kernel_regularizer=regularizers.l1(0.01)))
modelo_l1.add(Dense(np.unique(y_train).shape[0], activation="softmax"))

modelo_l1.compile(optimizer="sgd", loss="categorical_crossentropy",
               metrics=["accuracy"])

modelo_l1.fit(x_train_plano, y_train_one_hot, verbose=0, epochs=50, batch_size=1000)

acc_train = modelo_l1.evaluate(x_train_plano, y_train_one_hot)[1]
acc_test = modelo_l1.evaluate(x_test_plano, y_test_one_hot)[1]

resultados["regularizacion_l1"] = [acc_train, acc_test]
#%% Implementando dropout 
modelo_dropout = Sequential()
modelo_dropout.add(Dense(50, activation="relu", input_shape=(784,)))
modelo_dropout.add(Dense(250, activation="relu"))
# añadimos el DROPUOT y le decimos que apague un 20% de las neuronas en la capa que tiene 250 neuronas, esto se hace en cada iteracion
modelo_dropout.add(Dropout(0.2))# Se aplica a la capa previa
modelo_dropout.add(Dense(np.unique(y_train).shape[0], activation="softmax"))
# compilando...
modelo_dropout.compile(optimizer="sgd", loss="categorical_crossentropy",
               metrics=["accuracy"])

modelo_dropout.summary()
# ajustando el modelo
modelo_dropout.fit(x_train_plano, y_train_one_hot, verbose=0, epochs=50, batch_size=1000);
# obteniendo puntaje de exactitud en entrenamiento y prueba
acc_train = modelo_dropout.evaluate(x_train_plano, y_train_one_hot)[1]
acc_test = modelo_dropout.evaluate(x_test_plano, y_test_one_hot)[1]
#agregando resultado de regularización droup al diccionario
resultados["regularizacion_droup"] = [acc_train, acc_test]
#%% Implementando normalización en bloques
from keras.layers import BatchNormalization
modelo_bnorm = Sequential()
modelo_bnorm.add(Dense(50, activation="relu", input_shape=(784,)))
modelo_bnorm.add(Dense(250, activation="relu"))
# agregando normalización en bloques
modelo_bnorm.add(BatchNormalization()) # afecta a la capa de 250 neuronas
modelo_bnorm.add(Dense(np.unique(y_train).shape[0], activation="softmax"))
# compilando...
modelo_bnorm.compile(optimizer="sgd", loss="categorical_crossentropy",
               metrics=["accuracy"])

modelo_bnorm.summary()
# ajustando el modelo 
modelo_bnorm.fit(x_train_plano, y_train_one_hot, verbose=0, epochs=50, batch_size=1000)
# calculando puntaje de exactitud en entrenamiento y prueba
acc_train = modelo_bnorm.evaluate(x_train_plano, y_train_one_hot)[1]
acc_test = modelo_bnorm.evaluate(x_test_plano, y_test_one_hot)[1]
# guardando resultados.
resultados["batch_normalization"] = [acc_train, acc_test]
#%% Implementado regularización dropout  y normalización en bloques
modelo_bnorm_drop = Sequential()
modelo_bnorm_drop.add(Dense(50, activation="relu", input_shape=(784,)))
modelo_bnorm_drop.add(Dense(250, activation="relu"))
# agregando normalización en bloques. Afecta a la capa de 250 neuronas
modelo_bnorm_drop.add(BatchNormalization())
# añadimos el DROPUOT y le decimos que apague un 20% de las neuronas. afecta a la capa de 250 neuronas
modelo_bnorm_drop.add(Dropout(0.2))
modelo_bnorm_drop.add(Dense(np.unique(y_train).shape[0], activation="softmax"))

modelo_bnorm_drop.compile(optimizer="sgd", loss="categorical_crossentropy",
               metrics=["accuracy"])
# compilando el modelo
modelo_bnorm_drop.fit(x_train_plano, y_train_one_hot, verbose=0, epochs=50, batch_size=1000)
# calculando puntaje de exactitud
acc_train = modelo_bnorm_drop.evaluate(x_train_plano, y_train_one_hot)[1]
acc_test = modelo_bnorm_drop.evaluate(x_test_plano, y_test_one_hot)[1]
# guardando resultados
resultados["batch_normalization + dropout"] = [acc_train, acc_test]
#%% Analizamos el diccionario con los resultados, para esto lo converitmos en df
resultados = pd.DataFrame(resultados).T
resultados.columns = ["acc_train", "acc_test"]
resultados["pct_diff"] = 1 - (resultados.acc_test / resultados.acc_train)
# el resultado indica que debemos fijar en la puntación obtenida en el acc_test
# que es como se comportó con los datos que no conocia.