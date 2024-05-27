# -*- coding: utf-8 -*-
"""
Algoritmo para analizar datos de 
Calidad del aire - datos históricos diarios (1997-actualidad)
Fecha: 25/05/2024
@author: Luis A. García
"""
# Importando librerias necesarias
import os 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Accediendo a la ruta donde se descargó el archivo de datos
os.chdir('D:\\6. NEXER\\master\\Contenido\\4. Modulo 3 - Mineria del dato-Data mining\\tareas entregadas\\modulo 4.7 a(1)')
# Leyendo archivo *.csv
"""
Dataset:
    CO:   Monóxido de carbono
    NO:   Óxido nítrico 
    NO2:  Dióxido de nítrogeno
    O3:   Ozono 
    PM10:  Partículas en suspensión de diámetro menor a 10 micrómetros
    PM25: Partículas en suspensión de diámetro menor a 2.5 micrómetros
    SO2:  Dióxido de azufre
"""
file = pd.read_csv('calidad-del-aire-datos-historicos-diarios.csv',sep=';')
file.columns
# Verificando que columnas tienen datos faltantes
Colum_Datos_null =file.columns[file.isnull().any()]
# buscando filas que contienen datos == 999
#datos_malos = file[file.eq(999).any(axis=1)]
#eliminando fila con valores 999
#file=file[~file.eq(999).any(axis=1)]
# agrupando estaciones por latitud y longitud
estaciones = file.groupby(by =['Estación','Latitud','Longitud'],dropna=False).agg(
    cantidad_datos = ('Estación','count')
    ).reset_index()
#%% Procedemos a graficar las estaciones (opcional)
# importando librerias necesarias
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx
from pyproj import Transformer
# creando geodataframe desde el dataframe
geometry = [Point(xy) for xy in zip(estaciones['Longitud'], estaciones['Latitud'])]
gdf = gpd.GeoDataFrame(estaciones, geometry=geometry)

gdf = gdf.set_crs("EPSG:4326")

# Convirtiendo el CRS a Web Mercator para contextily
gdf = gdf.to_crs(epsg=3857)

# graficando
fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, marker='o', color='red', markersize=5)

# Añadiendo mapa base
"""
Podemos usar:
    ctx.providers.Esri.WorldStreetMap Estilo de mapa callejero de Esri
    ctx.providers.Esri.WorldTopoMap 
    ctx.providers.OpenStreetMap.Mapnik mapa estandar
"""
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# Transformando las coordenadas de los límites del gráfico de Web Mercator a WGS 84
transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326")
xmin, ymin, xmax, ymax = ax.get_xlim()[0], ax.get_ylim()[0], ax.get_xlim()[1], ax.get_ylim()[1]
x_ticks = ax.get_xticks()
y_ticks = ax.get_yticks()

# Transformando las coordenadas para las etiquetas de los ejes
x_labels = [transformer.transform(xx, ymin)[1] for xx in x_ticks]
y_labels = [transformer.transform(xmin, yy)[0] for yy in y_ticks]

# Configurando las etiquetas de los ejes
ax.set_xticklabels([f'{label:.2f}' for label in x_labels], fontname='Arial', fontweight='bold')
ax.set_yticklabels([f'{label:.2f}' for label in y_labels], fontname='Arial', fontweight='bold')

# Agregando grilla al gráfico
ax.grid(True)
#%% Agrupando estaciones por provincia
cant_provincia = file.groupby(by=['Provincia','Estación','Latitud','Longitud'],dropna = False).agg(
    registros_por_estacion = ('Estación','count')
    ).reset_index()
# calculando cantidad de estaciones por provincia
cant_est_prov = cant_provincia.groupby(by=['Provincia'],dropna=False).agg(
    cantidad_estaciones = ('Provincia','count')).reset_index()
#%% Solicitando los 10 primeros registros del datased
registros = file.head(10)
# Definiendo función que calcula el promedio 
def promedio_estacion(row,file):
    """ 
    row: es una fila de un DataFrame que contiene el nombre de la estación.
    por ejemplo 'Aranda de Duero'
    file: es el dataframe que contiene los datos de las estaciones
    """
    # obteniendo los datos de la columna estación
    nombre_estacion =row['Estación']
    # identificando los registros correspondientes a esa estacion
    condicion = (file['Estación']==nombre_estacion)
    # extrayendo todas las filas que perternecen a esa estacion
    data_estacion = file[condicion]
    # Calculando promedio para PM10
    PM10_mean = data_estacion['PM10 (ug/m3)'].mean()
    return PM10_mean
# creando columna 'PM10_mean' en el dataframe estaciones con los resultados de la función
estaciones['PM10_mean'] = estaciones.apply(promedio_estacion, axis=1, file=file)
#%%creando gráfico de barras forma 1
ax=estaciones.iloc[89:110].plot(kind='barh', x='Estación', y='PM10_mean', legend=False)
# Añadiendo etiquetas de valores en las barras
for rect in ax.patches:
    width = rect.get_width()
    ax.text(width, rect.get_y() + rect.get_height() / 2, f'{width:.2f}', ha='left', va='center', fontname='Arial', fontweight='bold')
plt.xlabel('Promedio de PM10 [ug/m3]', fontname='Arial', fontweight='bold',fontsize=14)
plt.ylabel('Estación', fontname='Arial', fontweight='bold',fontsize=14)
plt.title('Promedio de PM10 por Estación', fontname='Arial', fontweight='bold',fontsize=17)
plt.show()
# Personalizando etiquetas de ejes x e y
plt.xticks(fontname='Arial', fontweight='bold', fontsize=12)
plt.yticks(fontname='Arial', fontsize=9)
#%% creando gráfico de barras forma mejorada
# Calcular el número de grupos necesario según el tamaño deseado por grupo
import math
tamano_grupo = 22
num_grupos = math.ceil(len(estaciones) / tamano_grupo)

# Dividir el dataframe en grupos de 22 barras cada uno
for i in range(num_grupos):
    inicio = i * tamano_grupo
    fin = min((i + 1) * tamano_grupo, len(estaciones))
    subset_estaciones = estaciones.iloc[inicio:fin]
    
    # Crear el gráfico de barras horizontal para el grupo actual
    plt.figure(figsize=(10, 8))  # Tamaño del gráfico
    bars = plt.barh(subset_estaciones['Estación'], subset_estaciones['PM10_mean'], color='steelblue')
    
    # Agregar etiquetas en las barras
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height() / 2, f'{width:.2f}', ha='left', va='center', fontname='Arial', fontweight='bold')
    
    plt.xlabel('Promedio de PM10 [ug/m3]', fontname='Arial', fontweight='bold', fontsize=14)
    plt.ylabel('Estación', fontname='Arial', fontweight='bold', fontsize=14)
    plt.title('Promedio de PM10 por Estación', fontname='Arial', fontweight='bold', fontsize=17)
    plt.xticks(fontname='Arial', fontweight='bold', fontsize=12)
    plt.yticks(fontname='Arial', fontsize=9)
    
    # Guardar cada grupo como imagen PNG
    plt.savefig(f'grupo_{i + 1}.png', bbox_inches='tight')
    plt.close()
#%% Digramas de cajas y bigotes
# Procedemos a graficar las diagramas de cajas y bigotes
boxplotChl = file.boxplot(column=['CO (mg/m3)', 'NO (ug/m3)', 'NO2 (ug/m3)',
       'O3 (ug/m3)', 'PM10 (ug/m3)', 'PM25 (ug/m3)',
       'SO2 (ug/m3)'],
                    medianprops=dict(linestyle='-', linewidth=2, color='red'),
                    boxprops=dict(linewidth=2, color='blue'),
                    whiskerprops=dict(linewidth=2, color='black'),
                    flierprops=dict(marker='o', markersize=5, markerfacecolor='red', markeredgecolor='red'),
                    capprops=dict(linewidth=3, color='black'))
# Personalizando ejes
boxplotChl.set_xlabel('Variables', fontsize=20, fontweight='bold', labelpad=2)
boxplotChl.set_ylabel('medidas', fontsize=20, fontweight='bold')
boxplotChl.set_title('Diagramas de cajas y bigotes', fontsize=20, fontweight='bold')
boxplotChl.spines['top'].set_linewidth(1)  # Grosor del borde superior
boxplotChl.spines['right'].set_linewidth(1)  # Grosor del borde derecho
boxplotChl.spines['bottom'].set_linewidth(1)  # Grosor del borde inferior
boxplotChl.spines['left'].set_linewidth(1)  # Grosor del borde izquierdo
boxplotChl.tick_params(axis='both', direction='out', length=6)  # Dirección y longitud de los ticks
boxplotChl.xaxis.set_tick_params(width=2)  # Grosor de los ticks en el eje X
boxplotChl.yaxis.set_tick_params(width=2)  # Grosor de los ticks en el eje Y
#plt.savefig('diagrama_cajas_bigotes Chl.png', dpi=360)
#%%Grafico de densidad de distribución del O3
import seaborn as sns
# Creando el gráfico de densidad
plt.figure(figsize=(10, 6))
sns.kdeplot(file['O3 (ug/m3)'], shade=True, color='blue')
plt.title('Distribución de la variable O3', fontname='Arial', fontweight='bold',fontsize=17)
plt.xlabel('O3 [ug/m3]',fontname='Arial', fontweight='bold',fontsize=14)
plt.ylabel('Densidad',fontname='Arial', fontweight='bold',fontsize=14)
xmin, xmax = plt.xlim()
xticks = np.arange(start=np.floor(xmin / 50) * 50, stop=np.ceil(xmax / 50) * 50 + 1, step=50)
plt.xticks(xticks)
plt.savefig('grafico de densidad de distribucion.png', dpi=360)
#%% mapa de calor
# Seleccionando todas las columnas numericas
df_numerica = file.select_dtypes(['float64','int64'])
# seleccionando todas las columnas, excep latitud y longitud
df_num = df_numerica.loc[:,~df_numerica.columns.isin(['Latitud','Longitud'])]
# reemplanzando datos faltantes a np.nan
df_num = df_num.fillna(np.nan)
# calculando matriz de correlación sin tener los datos faltantes
Matriz_correlacion = df_num.dropna().corr()
# Crear el mapa de calor con seaborn
colores_mapa_calor = {
    'viridis': 'Viridis',
    'plasma': 'Plasma',
    'inferno': 'Inferno',
    'magma': 'Magma',
    'cividis': 'Cividis',
    'coolwarm': 'Coolwarm',
    'spring': 'Spring',
    'summer': 'Summer',
    'autumn': 'Autumn',
    'winter': 'Winter',
    'RdYlBu': 'Red-Yellow-Blue',
    'RdBu': 'Red-Blue',
    'PiYG': 'Pink-Green',
    'PRGn': 'Purple-Green',
    'BrBG': 'Brown-Blue-Green',
    'PuOr': 'Purple-Orange',
    'GnBu': 'Green-Blue',
    'BuGn': 'Blue-Green',
    'YlGnBu': 'Yellow-Green-Blue',
    'YlGn': 'Yellow-Green',
    'BuPu': 'Blue-Purple',
    'RdPu': 'Red-Purple',
    'YlOrRd': 'Yellow-Orange-Red',
    'OrRd': 'Orange-Red',
    'PuRd': 'Purple-Red'
}
# Configurar las propiedades de fuente globalmente
sns.set(font_scale=1.5, rc={'font.weight': 'bold', 'font.family': 'Arial'})

# Crear el heatmap
plt.figure(figsize=(10, 12))  # Ajustar el tamaño según sea necesario
heatmap = sns.heatmap(
    Matriz_correlacion, 
    xticklabels=Matriz_correlacion.columns, 
    yticklabels=Matriz_correlacion.columns, 
    cmap='inferno', 
    annot=True, 
    fmt='.2f',
    cbar=True,
    square=True,
    annot_kws={'size': 12, 'fontweight': 'bold', 'fontfamily': 'Arial'},  # Propiedades de fuente para las anotaciones
    linewidth=.5,
    linecolor='none'
)
# Ajustar las etiquetas del eje x para que sean más legibles
plt.xticks(fontsize=12, fontweight='bold', fontfamily='Arial')
plt.yticks(fontsize=12, fontweight='bold', fontfamily='Arial')
#nombres de ejes
plt.title('Matriz de correlación', fontname='Arial', fontweight='bold',fontsize=17)
# guardando imagen
plt.savefig('mapa de calor.jpg', dpi=360)
