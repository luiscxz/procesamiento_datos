# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 11:23:21 2024
Ejemplos de uso de gradientes en sklearn

@author: Luis A. García
"""
#Importando librerías necesarias
import pandas as pd
import os
import numpy as np
# accediendo a la ruta del archivo
os.chdir('D:\\3. Cursos\\11. Deep learning\\Deep_Learning_23-main\\tablas')
# leyendo archivo
Tabla = np.genfromtxt("C03_tabla.csv",delimiter=",")
coeficientes_objetivo = np.genfromtxt("C03_coeficientes_objetivo.csv",delimiter=",")
# convirtiendo array a dataframe
df = pd.DataFrame(Tabla)
df.columns = ["x1","x2","y"]
# definiendo variables independientes y objetivo
X = Tabla[:,0:2]
y = Tabla[:,2]
#%%  Gradiente estocastico en regresión lineal
# Importando gradiente estocastico para modelos de regresión lineal
from sklearn.linear_model import SGDRegressor
# definiendo modelo con un máximo de 10 iteraciones
estimador_sgd = SGDRegressor(max_iter=100)
# entrenando el modelo
estimador_sgd.fit(X, y)
# obteniendo predicciones del modelo
estimador_sgd.predict(X)[:10]
# procedemos a ver los coeficientes b1 y b2
estimador_sgd.coef_