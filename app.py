import streamlit as st
import keras
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
#import cv2
import time
from img_classification import *

input_shape = (224, 224)
image_types = ["png", "jpg"]

def main():
    st.title("Image Classification Model")
    #st.info(__doc__)
    st.header("Brain Tumor Classification Example")
    st.write('Select the state-of-the-art model to predict input image:')
    option_1 = st.checkbox('VGG16', key='vgg')
    option_2 = st.checkbox('ResNet50',key='resnet')
	#my_expander = st.beta_expander("ModelInfo")
	#with my_expander:
	#    "Details about this model" 
	#select_classifier = st.multiselect('Select the model',['VGG', 'ResNet','Inception'],['VGG','ResNet'])
	#st.write('you selected', select_classifier)
    results_sidebar = st.sidebar.checkbox('Results from trained model')
    if results_sidebar:
        st.sidebar.write("Results VGG16")
        st.sidebar.image("results_vgg.jpg")#results_side_vgg = 
        st.sidebar.write("")
        st.sidebar.write("")
        st.sidebar.write("Results Resnet50")
        st.sidebar.image('results_resnet.jpg') #results_side_resnet = 
    if not option_1 or option_2:
        st.write("Please select at least one of model to upload file for prediction.")
    if option_1 or option_2:
        uploaded_file = st.file_uploader("Choose a brain MRI image for classification as tumor or non-tumor...", type=image_types)
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            #image_display = image.resize(input_shape)
            #st.image(image_display, use_column_width=False)
            
            #progress_bar = st.empty()
            #progress = st.empty()
            if option_1:
                st.write("You have chosen VGG model for prediction, classifying. Please wait. ")
                #st.progress(25)
                st.spinner()
                label = binary_classifier_vgg(image, 'model_vgg.h5')
                #st.spinner()
                with st.spinner(text='Classification: Work in progress'):
                    time.sleep(3)
                    st.success('Done')
                if label == 1:
                    #st.image('')
                    st.write("Result: Apparently, this scan has a tumor. Please consult a doctor for prescription.")
                else:
                    #st.image('')	
                    #st.balloons()
                    st.write("Result: Seemingly, this looks like a healthy brain image. Take good care.")
                #st.success('Done')
            if option_2:
                st.write("")
                st.write("")
                st.write("")
                st.write('You have selected ResNet model for prediction, classifying. Please be patient.')
                st.spinner()
                label = binary_classifier_resnet(image,'model_resnet.h5')
                with st.spinner(text='Classification: Work in progress'):
                    time.sleep(3)
                    st.success('Done!')
                if label == 1:
                    st.write("Result: Seemingly, this scan caught a tumor. Please visit a doctor for prescription.")
                else:
                    #st.balloons()
                    st.write("Result: Apparently, no tumor exists yet. Take good care.")
            if option_1 and option_2:
                st.write("")
                st.write("")
                st.write("")
                st.info("The results from two models might differ. Please take medical assistance always.")        
main()