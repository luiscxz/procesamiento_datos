# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:42:42 2024

@author: Luis A. García
"""
# Importando librerías necesarias
import pandas as pd
import numpy as np
from plotnine import *
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from pandas.plotting import parallel_coordinates
from plotly.offline import plot
import os
#Accediendo a la ruta de los archivos
os.chdir('D:\\3. Cursos\\10. Estadistica multivariante\\Est_Mul_23-main\\tablas')
#leyendo archivo csv
iris = pd.read_csv('iris.csv')
#%% Visualizando histogramas y gráficos de densidad
# histograma absoluto sin tomar en cuentas las especies
# el histograma absoluta cuenta cuantos caen en cada paquete 
(ggplot(data=iris) +
  geom_histogram(mapping=aes(x="Sepal.Width"),
                 fill="red",
                 color="white",
                 bins=30) # bin es el número de barras
)
# Histograma relativo del ancho de sépalo sin distinguir especies incluyendo densidad
# el relativo indica que porcentaje del total cayó en cada paquete
(ggplot(data=iris) +
  geom_histogram(mapping=aes(x="Sepal.Width",
                             y="..density.."),
                 fill="red",
                 color="black") +
  geom_density(mapping=aes(x="Sepal.Width"),
               fill="red",
               color="black",
               alpha=0.5)
)
#%% Histograma de longitud de sépalo separado por especies
#seleccionando los valores de Petal.Length para todas las filas donde la especie es "setosa".
setosas = iris[iris['Species']== 'setosa']['Petal.Length']
#seleccionando los valores de Petal.Length para todas las filas donde la especie es "virginica".
virginica = iris[iris['Species']== 'virginica']['Petal.Length']
#seleccionando los valores de Petal.Length para todas las filas donde la especie es "versicolor".
versicolor = iris[iris['Species']== 'versicolor']['Petal.Length']
plt.hist(setosas,alpha=0.5)
plt.hist(virginica,alpha=0.5)
plt.hist(versicolor,alpha=0.5)
#%% Histograma de cada columna sin separar por especies
# Para esto convertimos el dataframe a formato largo
iris_ordenado = iris.melt(id_vars=['Species'],
                          var_name = 'Medida',
                          value_name='Valor')
(ggplot(data=iris_ordenado) +
      geom_histogram(mapping=aes(x="Valor"),
                     fill="red",
                     color="darkblue",
                     bins=30) +
      theme(
        panel_background = element_rect("black"), 
        panel_grid = element_blank(),
      ) +
      facet_wrap("~Medida") # divide el gráfico en subgráficos, dependiendo las especies 
      # contenidas en la variable Medida
)
#%%  Graficos de Densidad de la longitud de pétalo separado por especies
(ggplot(data=iris) +
  geom_density(mapping=aes(x="Petal.Length",
                           fill="Species"), # rellena con respecto  a las especies
               color="black",
               alpha=0.5)
)
# realizando Histograma con densidad de la longitud de pétalo separado por especies
sns.distplot(setosas)
sns.distplot(virginica)
sns.distplot(versicolor)
#%% Nubes de puntos
# Nube de puntos sin separar en especies

(ggplot(data=iris) +
  geom_point(mapping=aes(x="Sepal.Width",y="Sepal.Length"),color="red")
)
########### Nube de puntos separando en especies

(ggplot(data=iris) +
  geom_point(mapping=aes(x="Sepal.Width",y="Sepal.Length",color="Species")) +
  scale_color_manual(values=["red","orange","blue"])
)
#%% Graficos de dispersión con histogramas marginales
sns.jointplot(data=iris,x="Sepal.Length",y="Sepal.Width")
#Graficos de dispersión separando en especies con marginlaes y densidades
sns.jointplot(data=iris,x="Sepal.Length",y="Sepal.Width",hue="Species")
#%% Matriz de gráficos de dispersión (scatter plots) sin información extra
# forma 1
sns.pairplot(iris,hue="Species")
# forma 2 parecida a r
pair_plot = sns.pairplot(iris, hue="Species", diag_kind="kde", palette="husl", markers="o", plot_kws={'alpha':0.5})

# Ajustar el estilo del fondo y las líneas de la cuadrícula para cada eje
for ax in pair_plot.axes.flatten():
    ax.set_facecolor('#202020')  # Color del fondo del panel
    ax.grid(True, color='blue')  # Color de las líneas de la cuadrícula

# Eliminar los títulos de los ejes
for ax in pair_plot.axes.flatten():
    ax.set_xlabel('')
    ax.set_ylabel('')

# Mostrar la figura
plt.show()
#%% Matriz de gráficos de dispersión (scatter plots) con información extra

""" configurando el entorno de R_HOME
escribir en R studio el comando R.home() y copias el resultado
"""
os.environ['R_HOME'] = 'C:/PROGRA~1/R/R-4.4.1' 
# Importando librerias para usar R en python
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
# convirtiendo dataframe a objeto R
base = importr('base')
with localconverter(ro.default_converter + pandas2ri.converter):
  df_summary = base.summary(iris)

# Definiendo el código R que deseas ejecutar
r_code = '''
library(GGally)
library(ggplot2)

ggpairs(data=iris,
        mapping = aes(color=Species,alpha=0.01),
        columns=1:4) +
  theme(panel.background = element_rect("#202020"),
        panel.grid = element_line("blue"),
        axis.title = element_blank())
'''
#ro.r(r_code)
# Ejecutando el código R
print(ro.r(r_code))
#%% Verificando correlacione por grupos.
from scipy.stats import pearsonr
corr_by_species = iris.groupby('Species').corr(method='pearson')
#%% nube de puntos para 3 columnas
# creando objeto de nube de puntos tridimensional
nube_3d_sin_especies = px.scatter_3d(iris,x="Sepal.Length",y="Sepal.Width",z="Petal.Length")
# graficando al nuve de puntos
plot(nube_3d_sin_especies)
# creando objeto de nube de puntos tridimensional por especies
nube_3d_con_especies = px.scatter_3d(iris,x="Sepal.Length",y="Sepal.Width",z="Petal.Length",color="Species")
# graficando al nuve de puntos por especies
plot(nube_3d_con_especies)
#%% Densidades conjuntas de dos columnas (Mapa de contornos)
# procedemos a seleccionar todas las filas que contiene la clase setosa
setosa = iris[iris['Species']=='setosa']
# procedemos a seleccionar todas las filas que contiene la clase virginica
virginica = iris[iris['Species']=='virginica']
# realizando mapa de contorno para setosa
sns.kdeplot(data=setosa, x="Sepal.Width",y="Sepal.Length",
            cmap="Reds",fill=True,thresh=0.5)
# realizando mapa de contorno para virginica
sns.kdeplot(data=virginica, x="Sepal.Width",y="Sepal.Length",
            cmap="Oranges",fill=True,thresh=0.5)
#%% Mapa de correlaciones
# calculando correlaciones
correlaciones = (iris.loc[:,~iris.columns.isin(['Species'])]).corr()
# realizando mapa de calor
sns.heatmap(data=correlaciones,
            vmax=1,vmin=-1,
            cmap="Reds",
            linecolor="white",linewidth=0.5,
            square=True,
            annot=True,fmt=".2f")
#%% Procedemos a hacer el gráfico de coordenadas paralelas
import plotly.io as pio
# Configura Plotly para que use png
pio.renderers.default = 'svg'
iris['Species_id']= iris['Species'].replace({'setosa':1,'virginica':2,'versicolor':3})
# procedemos a establecer el orden en que queremos que aparezcan las columnas
fig = px.parallel_coordinates(iris, color="Species_id",
                              dimensions=['Sepal.Width', 'Sepal.Length', 'Petal.Width',
                                          'Petal.Length'],
                              color_continuous_scale=px.colors.diverging.Tealrose,
                              color_continuous_midpoint=2,)
fig.show()


#%% graficando coordenadas paralelas con R
r_code = '''
library(GGally)
library(ggplot2)

ggparcoord(data = iris,
           columns = c(1,4,3,2),
           #alphaLines = 0.1,
           #boxplot = TRUE,
           groupColumn = "Species",           
           showPoints = TRUE) +
  scale_color_brewer(palette = "Set2") +
  theme(panel.background = element_rect("#202020"),
        panel.grid = element_blank(),
        axis.title = element_blank()) 
'''
#ro.r(r_code)
# Ejecutando el código R
print(ro.r(r_code))