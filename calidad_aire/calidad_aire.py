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
# Accediendo a la ruta donde se descargó el archivo de datos
os.chdir('D:\\6. NEXER\\master\\Contenido\\4. Modulo 3 - Mineria del dato-Data mining\\tareas entregadas\\modulo 4.7 a(1)')
# Leyendo archivo *.csv
"""
Dataset:
    CO:   Monóxido de carbono
    NO:   Óxido nítrico 
    NO2:  Dióxido de nítrogeno
    O3:   Ozono 
    PM3:  Partículas en suspensión de diámetro menor a 10 micrómetros
    PM25: Partículas en suspensión de diámetro menor a 2.5 micrómetros
    SO2:  Dióxido de azufre
"""
file = pd.read_csv('calidad-del-aire-datos-historicos-diarios.csv',sep=';')
file.columns
# Verificando que columnas tienen datos faltantes
Colum_Datos_null =file.columns[file.isnull().any()]
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
#%%creando gráfico de barras
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