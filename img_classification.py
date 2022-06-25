#import keras
#import tensorflow.keras
from tensorflow import keras
from PIL import Image, ImageOps
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
#from tensorflow.keras.utils import img_to_array, load_img
##from typing import Callable, List, NamedTuple, Tuple
#from keras.applications import (InceptionV3, ResNet50, VGG16, imagenet_utils)
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet

input_shape = (224, 224)

def binary_classifier_vgg(img, model_file):
    model_vgg_loaded = keras.models.load_model(model_file)
    image = img.convert('RGB').resize(input_shape, Image.ANTIALIAS)
    image.load()
    image_array = np.array(image)
    expanded_image_array = np.expand_dims(image_array, axis=0)
    preprocessed_expanded_image_array = preprocess_input_vgg(expanded_image_array)
    pred = model_vgg_loaded.predict(preprocessed_expanded_image_array)
    output_label = [1 if x>0.5 else 0 for x in pred]
    return output_label[0]


def binary_classifier_resnet(img, model_file):
    model_resnet_loaded = keras.models.load_model(model_file)
    image = img.convert('RGB').resize(input_shape, Image.ANTIALIAS)
    image.load()
    image_array = np.array(image)
    expanded_image_array = np.expand_dims(image_array, axis=0)
    preprocessed_expanded_image_array = preprocess_input_resnet(expanded_image_array)
    pred = model_resnet_loaded.predict(preprocessed_expanded_image_array)
    output_label = [1 if x>0.5 else 0 for x in pred]
    return output_label[0]
