# -*- coding: utf-8 -*-
"""
% Código para validad correos electronicos usando:
    * Validación de sintaxis:
        Utiliza email_validator para validar el formato del correo electrónico.
    * Verificación del Dominio y Registros MX:
        Utiliza dnspython para verificar que el dominio tiene registros MX.
    * Verificación SMTP:
        Intenta establecer una conexión SMTP con el servidor del correo para 
        verificar la dirección.
    * Paralelización con concurrent.futures:
        Ejecuta las verificaciones en paralelo para mejorar el rendimiento.
@author: Luis A. García
"""
# importando librerias necesarias
import pandas as pd
import os
from email_validator import validate_email, EmailNotValidError
import dns.resolver
import concurrent.futures
import smtplib
# accediendo a ruta del archivo csv
os.chdir('D:\\6. NEXER\\master\\Contenido\\4. Modulo 3 - Mineria del dato-Data mining\\actividades\\modulo 4.6')
# leyendo archivo csv
file = pd.read_csv('4.6_Microactividad_datos_1.csv')
#%% Sección de validacion 
# Configuración del resolvedor DNS
resolver = dns.resolver.Resolver()
resolver.lifetime = 5  # Tiempo de espera de 5 segundos
resolver.nameservers = ['8.8.8.8', '8.8.4.4']  # Google DNS
# creando función para verificar si un dominio es válido y puede recibir correo
def verificar_smtp(email):
    try:
        # Estrayendo el dominio de la dirección de correo electrónico
        domain = email.split('@')[1]
        """Obteniendo lista de registros MX. Cada lista es un objeto que contiene
        información sobre un registro MX, incluyendo la prioridad y servidor
        de correo electrónico
        """
        mx_records = resolver.resolve(domain, 'MX')
        # accediendo al primer registro mx  para obtener el nombre del servidor de correo
        mx_record = str(mx_records[0].exchange)
        # creando instancia al servisor con un tiempo de espera de 10s
        server = smtplib.SMTP(timeout=10)
        # conectandose al servidor
        server.connect(mx_record)
        # iniciamos conversación SMTP con servidor de corro mediante el comando helo
        server.helo(server.local_hostname)
        # realizando envio de correo electrónico desde un correo ficticio que no espera respuesta
        server.mail('noreply@example.com')
        # comprovando si el correo a verificar acepta el correo que intento enviar
        code, message = server.rcpt(email)
        # cerrando conexión con el servidor
        server.quit()
        # si code ==250 indica que el correo electronico ha sido aceptado por el servidor para su entrega
        return code == 250
    except:
        return False
    
# Procedemos a crear la función de validación de correo que ará uso de la anterior función
def validar_email_completo(email):
    try:
        # verificando si el email tiene un formato correcto
        valid = validate_email(email)
        # oteniendo dominio asociado al correo electrónico
        domain = valid.domain
        # realizando segunda verificación mediante la funcion verificar_smtp
        if not verificar_smtp(email):
            return False, "Correo no válido según SMTP"
        return True, "Correo válido según SMTP"
    
    except EmailNotValidError:
        # Si se encuentra discrepancia en el formato se devuleve 'invalido'
        return 'invalido', "Email con formato inválido."

"""creando función que toma la columna del dataframe que contiene los
   correos electrónicos y los procesa utilizando la función validar_email_completo(email)
"""
def procesar_email(email):
    # el asterico significa que validar_email_completo devuleve una tupla
    return email, *validar_email_completo(email)

# Usar concurrent.futures para paralelizar la validación de correos electrónicos
# se usaran 10 hilos
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(procesar_email, file['email']))
# Desempaquetar los resultados y añadir al DataFrame
emails, validez, mensajes = zip(*results)
file['Email valido'] = validez
file['Mensaje Validacion'] = mensajes
grupos_emails = file.groupby(by = ['Email valido'],dropna=False).agg(
    cantidad = ('Email valido','count')
    ).reset_index()
malos = file[file['Email valido']==False]
#%% Validando correo mediante API
import requests

NEVERBOUNCE_API_KEY = 'tu_clave_de_API'  # Reemplaza 'tu_clave_de_API' con tu clave de API de NeverBounce

def verificar_neverbounce(email):
    try:
        response = requests.post(
            'https://api.neverbounce.com/v4/single/check',
            params={'key': NEVERBOUNCE_API_KEY},
            json={'email': email}
        )
        data = response.json()
        
        if data['result'] == 'valid':
            return True, "El correo electrónico es válido"
        else:
            return False, "El correo electrónico no es válido"
    except Exception as e:
        return False, f"Error al verificar el correo electrónico: {e}"

def validar_email_completo(email):
    try:
        # Verificación con NeverBounce
        is_valid, message = verificar_neverbounce(email)
        if not is_valid:
            return False, message

        return True, "El correo electrónico es válido"
    except Exception as e:
        return False, f"Error al validar el correo electrónico: {e}"

def procesar_email(email):
    return email, *validar_email_completo(email)
