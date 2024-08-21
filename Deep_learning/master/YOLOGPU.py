# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 13:32:32 2024

@author: anboo
"""

#%% Sección con yolo9
import os
os.chdir('D:\\13. proyectosDEPPLearning\\detección de fracturas\\FractureDetection.v1i.yolov9')
from ultralytics import YOLO
import torch
torch.cuda.set_device(0)
# Configura PyTorch para usar la primera GPU disponible
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device = 'cuda:0'
    print("Using GPU")
else:
    device = 'cpu'
    print("Using CPU")
# Cargando el modelo y moviéndolo a la GPU
model = YOLO('yolov9s.pt').to(device)
# entrenando el modelo durante 50 épocas, usando lotes de 5 imágenes, cada una redimensionada a 640x640 píxeles.
model.train(
    data='data.yaml', 
    epochs=50,            
    batch=-1, #modo automático para una utilización de la memoria del 60% GPU            
    imgsz=640,
    cache = 'disk',
    save_period = 5,   # Permite guardar las épocas        
    plots=True,
    patience=10, # regularización
    dropout =0.2 
)
#%% validando el modelo
# Carga el modelo entrenado desde el checkpoint más reciente
model = YOLO('path/to/latest_model.pt').to(device)  # Reemplaza con la ruta al último checkpoint guardado

# Realizar la validación con los datos de prueba
results = model.val(
    data='data.yaml',  # Archivo de configuración de datos que debe incluir el conjunto de prueba
    imgsz=640,         # Tamaño de imagen utilizado durante el entrenamiento
    batch=-1,          # Ajuste automático del tamaño del batch
    device=device      # Usa el dispositivo configurado (GPU o CPU)
)

# Acceder a las métricas
precision = results.box.mp  # Mean precision de todas las clases
recall = results.box.mr     # Mean recall de todas las clases
f1_score = 2 * (precision * recall) / (precision + recall)  # F1 Score

# AP para diferentes IoU thresholds
ap50 = results.box.ap50()  # AP para IoU=0.5
ap = results.box.ap()      # AP para IoU=0.5:0.95

# Métricas por clase
ap_class = results.box.maps  # mAP de cada clase
class_index = results.box.ap_class_index  # Índice de clase para cada AP

# Imprimir los resultados
print(f"Mean Precision (mp): {precision:.4f}")
print(f"Mean Recall (mr): {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")
print(f"AP@0.5: {ap50:.4f}")
print(f"AP@0.5:0.95: {ap:.4f}")
print(f"AP por clase: {ap_class}")
print(f"Índice de clases: {class_index}")
