# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 23:23:17 2024

@author: amand
"""

import os
import csv
import numpy as np

# Función para leer los vectores de características de una carpeta específica
def leer_vectores_caracteristicas(carpeta, output_path='nivel_1_base_de_datos'):
    # Ruta del archivo CSV de la carpeta
    csv_path = os.path.join(output_path, 'img_database', carpeta, f'{carpeta}.csv')
    
    # Lista para almacenar los vectores de características
    vectores_caracteristicas = []
    
    # Verificar si el archivo CSV existe
    if not os.path.exists(csv_path):
        print(f"No se encontró el archivo {csv_path}")
        return None
    
    # Leer el archivo CSV
    with open(csv_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Saltar el encabezado
        
        # Leer cada fila del archivo CSV
        for row in csv_reader:
            nombre_imagen = row[0]  # Nombre de la imagen
            vector_caracteristicas = np.array(row[1:], dtype=float)  # Convertir el vector a numpy array
            vectores_caracteristicas.append((nombre_imagen, vector_caracteristicas))
    
    return vectores_caracteristicas

# Ejemplo de uso
carpeta = 'n000001'  # Cambiar por la carpeta que quieras acceder
vectores = leer_vectores_caracteristicas(carpeta)

# Mostrar los vectores de características leídos
if vectores:
    for nombre_imagen, vector in vectores:
        print(f'Imagen: {nombre_imagen}, Vector de características: {vector}')
