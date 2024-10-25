import os
import csv
import numpy as np
from scipy.spatial import distance
from deepface import DeepFace

# Función para calcular las características de una imagen con DeepFace
def calcular_features_deepface(ruta_imagen):
    # Extraer las características (embedding) usando DeepFace
    resultado = DeepFace.represent(img_path=ruta_imagen,  enforce_detection=False)
    
    # El resultado es una lista de diccionarios, seleccionamos solo el embedding
    if resultado and 'embedding' in resultado[0]:
        return np.array(resultado[0]['embedding'])
    else:
        print(f"Error: No se encontró el embedding para la imagen {ruta_imagen}")
        return None

# Función para cargar las características de la base de datos desde un archivo CSV
def cargar_features_base_datos(csv_path):
    features_database = []
    with open(csv_path, 'r') as archivo_csv:
        lector_csv = csv.reader(archivo_csv)
        next(lector_csv)  # Saltar la primera fila (encabezado)
        for fila in lector_csv:
            nombre_imagen = fila[0]  # Primera columna es el nombre de la imagen
            vector_caracteristicas = np.array(fila[1:], dtype=float)  # Convertir las características en float, ignorando el nombre de la imagen
            features_database.append((nombre_imagen, vector_caracteristicas))
    return features_database

# Función para calcular diferentes métricas de similitud
def calcular_similitud(query_vector, db_vector, metric='euclidean'):
    if metric == 'euclidean':
        return distance.euclidean(query_vector, db_vector)
    elif metric == 'cosine':
        return 1 - distance.cosine(query_vector, db_vector)
    elif metric == 'chi2':
        return 0.5 * np.sum(((query_vector - db_vector) ** 2) / (query_vector + db_vector + 1e-10))
    else:
        raise ValueError(f"Métrica {metric} no soportada")

# Función para comparar la imagen de consulta con la base de datos
def comparar_img_query(query_image, base_datos_path, metric='euclidean'):
    # Calcular las características de la imagen de consulta
    query_vector = calcular_features_deepface(query_image)

    if query_vector is None:
        print("No se pudieron calcular las características de la imagen de consulta.")
        return

    similitudes = []

    # Recorrer las carpetas de la base de datos
    for carpeta in os.listdir(base_datos_path):
        carpeta_path = os.path.join(base_datos_path, carpeta)
        if os.path.isdir(carpeta_path):
            csv_path = os.path.join(carpeta_path, f'{carpeta}.csv')

            # Cargar los vectores de características de la carpeta actual
            features_database = cargar_features_base_datos(csv_path)

            # Comparar la imagen de consulta con cada imagen de la base de datos en esta carpeta
            for nombre_imagen_db, vector_db in features_database:
                similitud = calcular_similitud(query_vector, vector_db, metric)
                similitudes.append((nombre_imagen_db, similitud))

    # Ordenar las imágenes por similitud (mayor similitud o menor distancia)
    similitudes_ordenadas = sorted(similitudes, key=lambda x: x[1])

    # Mostrar los resultados (puedes ajustar cuántos mostrar)
    print(f"Top 5 imágenes más similares a {query_image}:")
    for img_name, similitud in similitudes_ordenadas[:5]:
        print(f"Imagen: {img_name}, Similitud: {similitud}")

# Ruta de ejemplo a la imagen de consulta y la base de datos
query_image_path = 'CBIR Faces Dataset 2024/n000010/n000010_11.jpg'
base_datos_path = 'nivel_3_base_de_datos/img_database'

# Llamar a la función de comparación con la métrica deseada (euclidiana, coseno o chi2)
comparar_img_query(query_image_path, base_datos_path, metric='euclidean')
