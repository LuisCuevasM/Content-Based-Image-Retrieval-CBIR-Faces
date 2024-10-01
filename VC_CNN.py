from tensorflow.keras.applications import VGG16, VGG19, DenseNet121, MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_densenet
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_input_mobilenet
from tensorflow.keras.models import Model
import numpy as np

def vc_cnn(path, CNN='VGG16'):
    if CNN == 'VGG16':
        fd = extract_features_vgg16(path)
    elif CNN == 'VGG19':
        fd = extract_features_vgg19(path)
    elif CNN == 'DenseNet121':
        fd = extract_features_densenet(path)
    elif CNN == 'MobileNetV2':
        fd = extract_features_mobilenet(path)
    else:
        fd = 0

    return fd

# Función para extraer características con VGG16
def extract_features_vgg16(img_path):
    # Cargar el modelo VGG16 preentrenado sin las capas densas (Fully Connected)
    base_model = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.output)

    # Cargar y preprocesar la imagen
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input_vgg16(img_data)

    # Extraer las características
    features = model.predict(img_data)
    # Aplanar las características para obtener un vector 1D
    features_flatten = features.flatten()

    return features_flatten

# Función para extraer características con VGG19
def extract_features_vgg19(img_path):
    # Cargar el modelo VGG19 preentrenado sin las capas densas (Fully Connected)
    base_model = VGG19(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.output)

    # Cargar y preprocesar la imagen
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input_vgg19(img_data)

    features = model.predict(img_data)
    features_flatten = features.flatten()

    return features_flatten

# Función para extraer características con DenseNet121
def extract_features_densenet(img_path):
    base_model = DenseNet121(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.output)

    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input_densenet(img_data)

    features = model.predict(img_data)
    features_flatten = features.flatten()

    return features_flatten

# Función para extraer características con MobileNetV2
def extract_features_mobilenet(img_path):
    base_model = MobileNetV2(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.output)

    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input_mobilenet(img_data)

    features = model.predict(img_data)
    features_flatten = features.flatten()

    return features_flatten



#Ejemplo de Uso
#vc=vc_cnn(r"C:\Users\amand\Downloads\CBIR_Faces_Dataset_2024\CBIR Faces Dataset 2024\n000493\n000493_2.jpg",'VGG16')
#print(vc)