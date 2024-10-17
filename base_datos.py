# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 21:50:51 2024

@author: amand
"""

import os
import cv2
import numpy as np
import csv
import Handcrafted_Method as HM

# Función para calcular el vector de características según el nivel de extracción
def calcular_vector_caracteristicas(ruta_imagen, nivel):
    if nivel == 1:
        hist = HM.handcrafted_method(ruta_imagen)
        return hist
    elif nivel == 2:
        pass  # Agregar lógica para nivel 2
    elif nivel == 3:
        pass  # Agregar lógica para nivel 3

# Función para guardar vectores de características en un archivo CSV
def guardar_en_csv(nombre_imagen, vector_caracteristicas, csv_writer):
    fila = [nombre_imagen] + vector_caracteristicas.tolist()
    csv_writer.writerow(fila)

# Función principal para procesar las imágenes
def procesar_imagenes(nivel, dataset_path='CBIR Faces Dataset 2024'):
    output_path=f'nivel_{nivel}_base_de_datos'
    # Recorrer las carpetas del dataset
    for carpeta in os.listdir(dataset_path):
        carpeta_original = os.path.join(dataset_path, carpeta)
        if os.path.isdir(carpeta_original):
            # Crear la carpeta destino
            carpeta_destino = os.path.join(output_path, 'img_database', carpeta)
            os.makedirs(carpeta_destino, exist_ok=True)

            # Crear el archivo CSV dentro de la carpeta correspondiente
            csv_path = os.path.join(carpeta_destino, f'{carpeta}.csv')
            with open(csv_path, mode='w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                # Escribir encabezado
                encabezado = ['imagen'] + [f'caracteristica_{i+1}' for i in range(26244)]  # Ajustar según el número de características
                csv_writer.writerow(encabezado)

                # Generar los nombres de las primeras 10 imágenes
                for i in range(1, 11):  # Iterar del 1 al 10
                    imagen_nombre = f'{carpeta}_{i}.jpg'
                    ruta_imagen = os.path.join(carpeta_original, imagen_nombre)

                    # Comprobar si la imagen existe
                    if not os.path.exists(ruta_imagen):
                        print(f"Imagen no encontrada: {ruta_imagen}")
                        continue

                    # Calcular el vector de características
                    vector = calcular_vector_caracteristicas(ruta_imagen, nivel)

                    # Guardar el vector de características en el archivo CSV dentro de la carpeta correspondiente
                    guardar_en_csv(imagen_nombre, vector, csv_writer)

# Llamar a la función para procesar las imágenes con el nivel deseado
#procesar_imagenes(nivel=1)