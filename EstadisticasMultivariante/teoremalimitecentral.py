# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 09:40:08 2024

Aplicación del teorema del límite cental, caso multivariante
@author: Luis A. García 
"""
# importando librerías necesarias
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
# accediendo a la ruta del archivo
os.chdir('D:\\3. Cursos\\10. Estadistica multivariante\\Est_Mul_23-main\\tablas')
# leyendo archivo iris
iris = pd.read_csv('iris.csv')
# para nuestro exprimento vamos a seleccionar 2 columnas
columnas = iris.loc[:,['Sepal.Width','Petal.Width']]
# vamos a seleccionar una muestra de 50 datos de forma aleatoria y le vamos a calcular la media, esto
# lo hacemos para 1000 muestras

medias = [columnas.sample(n=60, replace=False).mean() for i in range(1000)] 
# convirtiendo a dataframe
medias_df = pd.DataFrame(medias)
# graficando
sns.scatterplot(data=medias_df, x='Sepal.Width', y='Petal.Width', color='red')

