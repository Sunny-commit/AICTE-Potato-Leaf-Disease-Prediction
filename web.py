import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

def model_prediction(test_image):
    # Load model inside the function to avoid unnecessary memory usage
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    
    # Convert file object to image
    image = Image.open(test_image)
    image = image.resize((128, 128))  # Resize to match model input

    # Convert image to array
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension
    input_arr = input_arr / 255.0  # Normalize

    # Predict
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Sidebar
st.sidebar.title("Plant Disease System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Disease Recognition'])

# Load and display header image
img = Image.open('Diseases.png')
st.image(img, use_column_width=True)

# Home Page
if app_mode == 'Home':  
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>", 
                unsafe_allow_html=True)

# Disease Recognition Page
elif app_mode == 'Disease Recognition':
    st.header('Plant Disease Detection System for Sustainable Agriculture')

    # File Uploader
    test_image = st.file_uploader('Choose an image:', type=['jpg', 'png', 'jpeg'])

    # Show Image Button
    if test_image is not None:
        if st.button('Show Image'):
            st.image(test_image, use_column_width=True)

        # Predict Button
        if st.button('Predict'):
            st.snow()
            result_index = model_prediction(test_image)
            class_name = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
            st.success(f'Model is predicting it as: **{class_name[result_index]}**')
