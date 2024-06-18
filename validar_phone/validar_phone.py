# -*- coding: utf-8 -*-
"""
Created on Fri May 31 16:04:10 2024

@author: anboo
"""
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
#%% validando 
import phonenumbers
from phonenumbers import NumberParseException, is_possible_number, is_valid_number, format_number, PhoneNumberFormat

def verify_phone(row):
    try:
        phone = str(row['phone']).strip()
        pais = str(row['country_code']).strip()
        
        number = phonenumbers.parse(phone, pais)
        
        if not is_possible_number(number):
            return {"valid": False, "formatted": None, "error": "Número imposible"}
        
        if not is_valid_number(number):
            return {"valid": False, "formatted": None, "error": "Número inválido"}
        
        num_formateado = format_number(number, PhoneNumberFormat.E164)
        return {"valid": True, "formatted": num_formateado, "error": None}
    
    except NumberParseException as e:
        return {"valid": False, "formatted": None, "error": str(e)}

# Aplicando la validación
resultados = data_clear.apply(verify_phone, axis=1)
data_clear['Valido'] = resultados.apply(lambda x: x["valid"])
data_clear['Numero formateado'] = resultados.apply(lambda x: x["formatted"])
data_clear['Error'] = resultados.apply(lambda x: x["error"])

# Estadísticas de resultados
num_validos = data_clear['Valido'].sum()
num_invalidos = len(data_clear) - num_validos
print(f"Números válidos: {num_validos}, Números inválidos: {num_invalidos}")