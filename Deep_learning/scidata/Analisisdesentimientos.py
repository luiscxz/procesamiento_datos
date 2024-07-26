# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 10:00:36 2024
 Análisis de sentimiento:
     Estos análisis consisten en dado un texto, predecir si el texto es un texto
     con tono positivo o negativo
@author: Luis A. Garía
"""
# importando librerías necesarias
import pandas as pd
import numpy as np
import os
# accediendo a ruta donde están los datos
os.chdir('D:\\3. Cursos\\11. Deep learning\\Deep_Learning_23-main\\tablas')
#leyendo archivo, que contiene tweets
tweets_corpus = pd.read_csv("texto_extendido.csv",encoding = "latin-1")
# renombrando columna polarity del dataframe
tweets_corpus = tweets_corpus.rename(columns={"polarity":"value"})
# verificando los tipos de columnas que tiene el df
tweets_corpus.info()
# contando las clases que tenemos
tweets_corpus['value'].value_counts()
# convirtiendo  los datos de la columna 'value' a variable dummy
polaridades = pd.get_dummies(tweets_corpus['value']).astype(int)
# convirtiendo dataframe a array
polaridades_one_hot = polaridades.values
# creando diccionario que asigna valor a los nombres de las polaridades
polaridades_dict = dict(zip(range(6), polaridades.columns))
polaridades_dict
#%% Procesamiento de texto
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
"""
Tokenizer: divide el texto en palabras. Ejemplo: "Hola, ¿cómo estás?", el resultado
    sería:
         ["Hola", "¿cómo", "estás", "?" ]
pad_sequences: Normaliza las secuencias para que todas tengan la misma longitud.
"""
# definiendo número máximo de palabras únicas que el tokenizer considerará.
max_palabras = 1500 
# creando objeto tokenizador
tokenizer = Tokenizer(num_words=max_palabras)
# convirtiendo columna 'content' a str
col_content = tweets_corpus['content'].astype('str')
# Ajustando el tokenizador
tokenizer.fit_on_texts(col_content.values)
# convirtiendo los textos en secuencias de enteros
X = tokenizer.texts_to_sequences(col_content.values)
#normalizando las secuencias
X = pad_sequences(X)
# consultando la codificación de las palabras
tokenizer.word_index
""" Crenado mapa inverso de la lista de palabras, esto nos permite reconstruir un 
    tweet en función de su versión vectorizada. 
    esto nos muestra el código y la palabra
"""
mapa_inverso = dict(map(reversed, tokenizer.word_index.items()))
mapa_inverso
#%% Ejemplo de armar tweets en función de su versión vectorizada
""" Vamos a ver un ejemplo de armar la palabra en función de su versión vectorizada
    Tomamos el primer tweets y vemos su versión vectorizada
"""
tweets_corpus['content'].values[0]
#versión vectorizada
X[0]
# armar la palabra en función de su versión vectorizada 
[mapa_inverso[i] for i in X[0] if i in mapa_inverso] 
#%% Implementación de la red neuronal por recurrencia
"""
Como capa de entrada vamos a usar la capa Embedding. Es una capa que se encarga
de transformar una matriz de texto (donde cada número representa una palabra), 
en una matriz que representa las relaciones entre las frases en función de sus palabras.

Digamos que un Embedding (word embedding) es una forma mejor de representar 
texto (captura mejor la información).
"""
from keras import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU
# definiendo número de clases
numero_clases = 6
input_length=X.shape[1]
# creando modelo secuencial
modelo_sentimiento = Sequential()
""" añadiendo capa Embedding donde:
    max_palabras: Es el número máximo de palabras o índices que la capa Embedding manejará.
    128: Este es el tamaño de cada vector de incrustación. Cada palabra (o índice) 
        se representará como un vector de 128 dimensiones.
    input_length: son las caracteristicas de entrada (número de columnas)   
"""
modelo_sentimiento.add(Embedding(max_palabras, 128, input_length=input_length))
""" Añadiendo capa Long Short-Term Memory que es capaz de aprender dependencias 
    a largo plazo en secuencias de datos. 
    256: Este es el número de unidades de memoria en la capa LSTM. 
    Cada unidad de memoria puede capturar características de la secuencia de entrada.
    
    dropout=0.2: Es la regularización que apaga neuronas en cada iteración
    
    recurrent_dropout=0.2: Esta es la tasa de dropout para las conexiones 
    recurrentes. Similar al dropout, pero se aplica a las conexiones dentro de 
    la LSTM que transportan información de una unidad de tiempo a la siguiente.
    
    return_sequences=True: Este parámetro indica que la capa LSTM debe devolver
    la secuencia completa de salidas para cada entrada en la secuencia
"""
modelo_sentimiento.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
modelo_sentimiento.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))
#modelo_sentimiento.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))
# añadiendo capa densa con 6 neuronas y activación multiclase
modelo_sentimiento.add(Dense(numero_clases, activation='softmax'))
# compilando el modelo
modelo_sentimiento.build(input_shape=(None, input_length))
modelo_sentimiento.compile(loss = 'categorical_crossentropy', optimizer='adam',
                           metrics = ['accuracy'])
print(modelo_sentimiento.summary())
#%% Separando datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split
Y = polaridades_one_hot
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20)
#%% aplicando EarlyStopping para monitorizar la presición y parar cuando esta no mejora
from keras.callbacks import EarlyStopping
batch_size = 256 #muestras
early_stop = EarlyStopping(monitor='accuracy', min_delta=0.00001, patience=5, verbose=1)
#ajustando el modelo
modelo_sentimiento.fit(X_train, Y_train, epochs=50, batch_size=batch_size, verbose=1,
                      callbacks=[early_stop]);
# evaluando modelo con datos de test
loss, precision_test = modelo_sentimiento.evaluate(X_test, Y_test)
precision_test
#%% Nos llega una nueva frase y queremos predecir su polaridad
nueva_frase = "Qué bonito es el amor"
nueva_frase_tokenizada = tokenizer.texts_to_sequences([nueva_frase])
nueva_frase_tokenizada_pad = pad_sequences(nueva_frase_tokenizada, maxlen=42)
nueva_frase_tokenizada
nueva_frase_tokenizada_pad
# revisando con el dicciónario
[mapa_inverso[i] for i in [63, 16, 3, 725] if i in mapa_inverso] 
#obteniendo predicción
predictions = modelo_sentimiento.predict(nueva_frase_tokenizada_pad)
# obteniendo clase a la que pertenece
predicted_class = np.argmax(predictions)
# mostrando clase
polaridades_dict[predicted_class]