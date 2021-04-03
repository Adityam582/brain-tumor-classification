import keras
from PIL import Image, ImageOps
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
##from typing import Callable, List, NamedTuple, Tuple
#from keras.applications import (InceptionV3, ResNet50, VGG16, imagenet_utils)
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet

input_shape = (224, 224)

# image = image.resize(input_shape)
# image = img_to_array(image)
# image = np.expand_dims(image, axis=0)
#image = self.preprocess_input_func(image)

# preprocess_input_func=inception_v3.preprocess_input
# decode_predictions_func=inception_v3.decode_predictions
# preprocess_input_func=resnet.preprocess_input
# decode_predictions_func=resnet.decode_predictions
# preprocess_input_func=vgg19.preprocess_input
# decode_predictions_func=vgg19.decode_predictions

def binary_classifier(img, weights_file):
    # Load the model
    model = keras.models.load_model(weights_file)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    #    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    normalized_image_array = image_array.astype(np.float32)/255

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    return np.argmax(prediction) # return position of the highest probability

def binary_classifier_vgg(img, weights_file):
    model_vgg_loaded = keras.models.load_model(weights_file)
    image = img.convert('RGB').resize(input_shape, Image.ANTIALIAS)
    image.load()
    image_array = np.array(image)
    x = np.expand_dims(image_array, axis=0)
    x = preprocess_input_vgg(x)
    pred = model_vgg_loaded.predict(x)
    output_label = [1 if x>0.5 else 0 for x in pred]
    return output_label[0]


def binary_classifier_resnet(img, weights_file):
    model_resnet_loaded = keras.models.load_model(weights_file)
    image = img.convert('RGB').resize(input_shape, Image.ANTIALIAS)
    image.load()
    image_array = np.array(image)
    x = np.expand_dims(image_array, axis=0)
    x = preprocess_input_resnet(x)
    pred = model_resnet_loaded.predict(x)
    output_label = [1 if x>0.5 else 0 for x in pred]
    return output_label[0]
