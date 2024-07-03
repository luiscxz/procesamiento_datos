# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 08:33:01 2024
Scrapear con selenium EDGE
@author: Luis A. García
"""
# importando librerias necesarias
import os
import selenium
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# accediendo al directorio donde tenemos el webdriver
os.chdir('D:\\3. Cursos\\0. web scraping python\\Python Selenium con XPATH')
path_driver = os.chdir('D:\\3. Cursos\\0. web scraping python\\Python Selenium con XPATH')
# Configurando las opciones de Edge
options = Options()
# Configurando el servicio para Edge
service = Service(executable_path=path_driver)
# Procedemos a iniciar nuestro objetivo
driver = webdriver.Edge(service=service, options=options)
# establecemos la url donde queremos sacar los datos
url = "https://onefootball.com/es/partido/2469874"
#accediendo a la url en modo pantalla completa
driver.maximize_window()
driver.get(url)
"""procedemos a obtener el texto Women:
    Se encuentra en las etiquetas:
        nav
        ol
        li
        a
"""
#%% Procedemos a dar clic en el boton correspondiente Ucrania
# Esperar hasta que el botón esté presente
wait = WebDriverWait(driver, 10)
button = wait.until(EC.element_to_be_clickable((By.XPATH, '//button[.//span[text()="Ucrania"]]')))
# Hacer clic en el botón
button.click()
""" Obteniendo formación de Ucrania
obteniendo los elementos que estan dentro de la clase li cuyo atributo class contiene
la frase "MatchLineupFormation_row__cbgug"
"""
formacionU = driver.find_elements(By.XPATH, '//li[contains(@class, "MatchLineupFormation_row__cbgug")]//a/p')
# extrayendo los elementos de la lista
jugadoresU = [element.text for element in formacionU]

#%% Procedemos a dar clic en el boton correspondiente a Rumania
wait = WebDriverWait(driver, 10)
button = wait.until(EC.element_to_be_clickable((By.XPATH, '//button[.//span[text()="Rumania"]]')))
# Hacer clic en el btón
button.click()
""" Obteniendo formación de Rumania
obteniendo los elementos que estan dentro de la clase li cuyo atributo class contiene
la frase "MatchLineupFormation_row__cbgug"
"""
formacionR = driver.find_elements(By.XPATH, '//li[contains(@class, "MatchLineupFormation_row__cbgug")]//a/p')
# extrayendo los elementos de la lista
jugadoresR = [element.text for element in formacionR]
#%% Procedemos a obtener los resultados de los ultimos partidos jugados del equipo Rumania
"""
Obteniendo los elementos que estan dentro de la clase ul cuyo atributo class 
contiene la frase "FormGuide_list__qSdNh"
"""
p_texts = [p.text for p in driver.find_elements(By.XPATH, '//ul[contains(@class, "FormGuide_list__qSdNh")]//p')]
# Obtener los atributos alt de los elementos <img>
img_alts = [img.get_attribute('alt') for img in driver.find_elements(By.XPATH, '//ul[contains(@class, "FormGuide_list__qSdNh")]//img')]

    


