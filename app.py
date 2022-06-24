import streamlit as st
#import keras
from tensorflow import keras
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import time
from img_classification import *

input_shape = (224, 224)
image_types = ["png", "jpg"]

st.set_page_config(
     page_title='Brain Tumor Classification Deep Learning Model',
     layout="wide",
     #initial_sidebar_state="",
)

def main():
    st.title("Image Classification Model")
    st.header("Deep transfer learning model with State-Of-The-Art models")
    st.write('Select the model to predict input image:')
    option_1 = st.checkbox('VGG16', key='vgg')
    option_2 = st.checkbox('ResNet50',key='resnet')
    results_sidebar = st.sidebar.checkbox('Results from trained model')
    if results_sidebar:
        st.sidebar.write("Results VGG16")
        st.sidebar.image("results_vgg.jpg") 
        st.sidebar.write("")
        st.sidebar.write("")
        st.sidebar.write("Results Resnet50")
        st.sidebar.image('results_resnet.jpg')
    if not option_1 or option_2:
        st.write("Please select at least one model to upload file for prediction.")
    if option_1 or option_2:
        uploaded_file = st.file_uploader("Choose a brain MRI image for classification as tumor or non-tumor...", type=image_types)
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            #image_display = image.resize(input_shape)
            #st.image(image_display, use_column_width=False)
            if option_1:
                st.write("You have chosen VGG model for prediction, classifying now. ")
                st.spinner()
                with st.spinner(text='Classification: Work in progress. Please wait.'):
                    label = binary_classifier_vgg(image, 'model_vgg.h5')
                    #time.sleep(1)
                    st.success('Done.')
                if label == 1:
                    st.write("Result: Apparently, this scan has a tumor. Please consult a doctor for prescription.")
                else:
                    st.write("Result: Seemingly, it shows no traces of a tumor presence. Take good care.")
            if option_2:
                st.write("")
                st.write("")
                st.write("")
                st.write('You have selected ResNet model for prediction, classifying now.')
                st.spinner()
                with st.spinner(text='Classification: Work in progress. Please be patient.'):
                    label = binary_classifier_resnet(image,'model_resnet.h5')
                    #time.sleep(1)
                    st.success('Done.')
                if label == 1:
                    st.write("Result: Seemingly, this scan caught a tumor. Please visit a doctor for prescription.")
                else:
                    st.write("Result: Apparently, no tumor exists yet. Take good care.")
            if option_1 and option_2:
                st.write("")
                st.write("")
                st.write("")
                st.info("The results from two models might differ. Please rely only on professional medical assistance. This page respects user data privacy and does not store any data such as images or results.")        
main()
