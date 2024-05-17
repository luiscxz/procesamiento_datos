# -*- coding: utf-8 -*-
"""
Algorimo para actividad DATA MANAGEMENT

@author: Luis A. García
"""
# Importando librerias necesarias
import os 
import glob
import pandas as pd
# Accediendo a la ruta donde están los archivos
os.chdir('D:\\6. NEXER\\master\\Contenido\\4. Modulo 3 - Mineria del dato-Data mining\\actividades\\modulo 4.6')
# detectando archivos .csv
csv = glob.glob('*.csv')
#%% Exploración de datos y concatenando archivos csv
# Detectando las columnas de los archivos
columnas={}
data = pd.DataFrame()
for i in range(len(csv)):
    # leyendo archivo csv
    file = pd.read_csv(csv[i])
    # almacenando nombre de columnas en el diccionario
    columnas[i]=file.columns.tolist()
    # concatenando dataframe
    data = pd.concat([data,file],axis=0)
#%%
# analizando los grupos de datos que tenemos por nombre apellido y dirección
grupo = data.groupby(by=['first_name','last_name','email','phone','country']).agg(
    cantidad = ('phone','count')
    )
# Ordenar el DataFrame resultante de mayor a menor según la columna 'cantidad'
grupo= grupo.sort_values(by='cantidad', ascending=False)
# analizando si exiten líneas con correos duplicados
linea_duplicada= data[data.duplicated(subset=['email','phone'], keep=False)]
# creando copia del dataframe data y eliminando datos duplicados
data_clear = data.copy()
# eliminando datos duplicados, pero conservando una instancia de ellos
data_clear.drop_duplicates(subset=['email','phone'], keep='first', inplace=True)
#%% buscando de datos faltantes 
#buscando filas donde falten datos
filas_null = data_clear[data_clear.isnull().any(axis =1)]
# buscando filas con valores -9999
filas_9999 = data_clear[data_clear.eq(-9999).any(axis=1)]
# buscando columnas donde falten datos
colum_null = data_clear.columns[data_clear.isnull().any()]
#%% Buscando correo electronicos mal formateados, usaremos la libreria email-validor
# para validar el formato del correo y verificar la existencia del dominio.
from email_validator import validate_email, EmailNotValidError
# creando función para usar validate_email
def validar_email(email): # email es una columna de un df que tenga correo electrónicos
    # manejo se excepciones de errores
    try:
        valid = validate_email(email)
        # verificando si el dominio acepta correos electrónicos
        domain_accepts_emails = True # inicialmente suponemos que el dominio acepta correo
        if domain_accepts_emails:
            # Devuelve True si el domino acepta email
            return True, f" El dominio {valid.domain} acepta email"
        else :
            # Devueleve False si el dominio no acepta email
            return False, f" El dominio {valid.domain} no acepta email"
    except EmailNotValidError:
        # si el email no tiene formato válido devuelve None
        return None, "Email con formato inválido."
# Creando columna en el dataframe que contiene los resultados de la validacion
data_clear['Email valido'], data_clear['Mensaje Dominio'] = zip(*data_clear['email'].apply(validar_email))
#data_clear['Email valido'] = data_clear['email'].apply(validar_email)
#%%