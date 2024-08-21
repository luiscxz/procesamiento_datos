# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:52:18 2024
Identificación de coches (en video.mp4) mediante modelo YOLO
Comandos de instalación:
    conda install pytorch::pytorch
    conda install -c conda-forge ultralytics
@author: Luis A. García
"""
# Importando librerías necesarias
import os
from ultralytics import YOLO
# accediendo a la ruta donde está el video
os.chdir('D:\\6. NEXER\\master\\Contenido\\6. Deep Learning\\tareas\\modulo 6.5 act2')
# cargando modelo YOLO
model = YOLO('yolov9m.pt')
results = model. track(source='Coches.mp4', show=True)
#%%
import cv2
# Configurar la captura del video
cap = cv2.VideoCapture('Coches.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model.track(source=frame, show=False)
    
    # Procesar las detecciones
    for detection in results[0].boxes:
        x1, y1, x2, y2 = map(int, detection.xyxy[0])
        label = detection.cls[0]
        confidence = detection.conf[0]
        
        # Dibuja el cuadro de detección
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Dibuja la etiqueta y la probabilidad
        text = f"{model.names[int(label)]} ({confidence:.2f})"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()