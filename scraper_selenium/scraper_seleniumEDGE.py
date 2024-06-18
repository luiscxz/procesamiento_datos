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
#%% Procedemos a dar clic en los botones correspondiente a el equipo
# Esperar hasta que el botón esté presente
wait = WebDriverWait(driver, 10)
button = wait.until(EC.element_to_be_clickable((By.XPATH, '//button[.//span[text()="Ucrania"]]')))
# Hacer clic en el botón
button.click()

#%% Obteniendo formación de Rumania
#texto1 = driver.find_element(By.XPATH,'//nav/ol/li/a[contains(text(),"Wo")]').text
# procedemos a obtener los textos de todos los menu
jugadores= driver.find_elements(By.XPATH,'//li/a/p')
# extrayendo los elementos de la lista
menu_texts = [element.text for element in jugadores]
# extrayendo los elementos de la lista

    


