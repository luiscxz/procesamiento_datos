# -*- coding: utf-8 -*-
"""
Algorimo para actividad DATA MANAGEMENT

@author: Luis A. García
"""
# Importando librerias necesarias
import os 
import glob
import pandas as pd
import numpy as np
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
        return False, "Email con formato inválido."
# Creando columna en el dataframe que contiene los resultados de la validacion
data_clear['Email valido'], data_clear['Mensaje Dominio'] = zip(*data_clear['email'].apply(validar_email))
#data_clear['Email valido'] = data_clear['Email valido'].replace({None: np.nan})
# obteniendo los gurpos formados por correo validos
grupos_emails = data_clear.groupby(by = ['Email valido'],dropna=False).agg(
    cantidad = ('Email valido','count')
    ).reset_index()
data_clear.to_csv('Copia dataframe data_clear')
#%% Mapeando pasises
import pycountry
grupo_paises = data_clear.groupby(by='country',dropna=False).agg(
    paises = ('country','count')
    ) 
# procedemos a obtener los códigos de región de cada pais (ISO 3166-1 alfa-2.)
def code_region(nombre_pais):
    # manejo de errores
    try:
        # buscando los paises del dataframe en la base de datos de pycountry
        country = pycountry.countries.lookup(nombre_pais)
        # si el pais es encontrado en la base de datos, retorna el código del pais en 2 letras
        return country.alpha_2
    # si encuentra un error devuelve
    except LookupError:
        return False
# aplicando la función code_regio a la columna country y creamos una nueva columna con los 
# código obtenidos
data_clear['country_code'] = data_clear['country'].apply(code_region)
# buscando lineas que retornaron False
lineas_false = data_clear[data_clear['country_code'] == False]
# Dado que Russia no le pudo encontrar el codigo procedemos a ponerle su codigo correspondiente
data_clear.loc[data_clear['country']== 'Russia','country_code'] ='RU'
"""
Palestinian Territory = 'PS'
Swaziland = 'SZ'
Democratic Republic of the Congo = 'CD'
Macedonia = 'MK'
"""
data_clear.loc[data_clear['country']== 'Palestinian Territory','country_code'] ='PS'
data_clear.loc[data_clear['country']== 'Swaziland','country_code'] ='SZ'
data_clear.loc[data_clear['country']== 'Democratic Republic of the Congo','country_code'] ='CD'
data_clear.loc[data_clear['country']== 'Macedonia','country_code'] ='MK'

#%% Procedemos a verificar los numeros de telefonos 
# importando librerias necesarias

import phonenumbers
from phonenumbers import NumberParseException, is_possible_number, is_valid_number, format_number, PhoneNumberFormat
# creando función para validar y formatear los numeros de teléfono
""" Row representa una columna específica de un dataframe que está
siendo procesada por la función apply
"""
def verify_phone(row): 
    # manejo excepciones 
    try:
        # obteniendo el teléfono y el país
        phone = row['phone']
        pais = row['country_code']
        # le pasamos el numero de telefono, None significa que el código del paíes debe ser dectatado automaticamente
        number = phonenumbers.parse(phone,pais) # parse(telefono, pais) ejemplo pais ='US'
        # verificando si es un número posible
        if not is_possible_number(number):
            return False, "Número imposible"
        # Verificando si es un número válido
        if not is_valid_number(number):
            return False, "Número invalido"
        # formatenado número a formato E164
        num_formateado = format_number(number,PhoneNumberFormat.E164)
        return True, num_formateado
    except NumberParseException as e:
        return False, str(e)
# Aplicando validación a los numeros de telefonos contenidos en la columna phone
data_clear['Valido'], data_clear['Numero formateado'] = zip(*data_clear.apply(verify_phone, axis=1))
# procedemos a ver cuantos numeros salieron validos y cuantos no
phone_validos = data_clear.groupby(by='Valido',dropna = False).agg(
    contantidad = ('Valido','count')).reset_index()
data_clear =data_clear.rename(columns={'country_code':'countrycode'})
#%% Procedemos a verificar las ciudades


#%% procedemos a armar el  dataframe limpio 
# seleccionando todas las filas donde Email valido y Valido sea igual a True
microactividad_clear = data_clear[(data_clear['Email valido']== True) & (data_clear['Valido']== True)]

data_clear=data_clear.drop('is_valid',axis=1)