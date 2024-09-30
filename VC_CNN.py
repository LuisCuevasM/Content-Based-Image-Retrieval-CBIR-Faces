from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
from tensorflow.keras.models import Model
import numpy as np

def vc_cnn(path, CNN='VGG16'):
    if CNN == 'VGG16':
        fd = extract_features_vgg16(path)
    elif CNN == 'VGG19':
        fd = extract_features_vgg19(path)
    else:
        fd = 0

    return fd

def extract_features_vgg16(img_path):
    # Cargar el modelo VGG16 preentrenado sin las capas densas (Fully Connected)
    base_model = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.output)

    # Cargar y preprocesar la imagen
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)  # Añadir una dimensión extra (batch)
    img_data = preprocess_input_vgg16(img_data)  # Preprocesar la imagen como lo hace VGG16

    # Extraer las características
    features = model.predict(img_data)
    # Aplanar las características para obtener un vector 1D
    features_flatten = features.flatten()

    return features_flatten

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