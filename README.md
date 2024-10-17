# Content-Based-Image-Retrieval-CBIR-Faces
El objetivo de este proyecto es desarrollar e implementar un algoritmo de búsqueda de imágenes de rostros similares.


El archivo base_datos.py genera los vectores característica usando las funciones de niveles anteriores 
cosas a cosiderar :
* Actualmente esta implementado solo el nivel 1
* Utiliza las 10 primeras imágenes, la 11 se usara como img_query (imagen de consulta)
* Se puede leer los vectores guardados en csv con Lector_vc.py, este puede imprimir el vector de caracteristicas y el nombre de la imagen original. No creo que sea necesario esto último pero me ayudo a comprobar de que imagenes se guaradaron en cada archivo.

* Se puede correr directo el archivo, no es necesario cambiar ninguna ruta
