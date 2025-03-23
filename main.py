import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --------------------------- TensorFlow Model Prediction ---------------------------
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')  # Load Model
    image = Image.open(test_image).resize((128, 128))  # Open and Resize Image
    input_arr = np.array(image) / 255.0  # Normalize Image
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert to Batch Format
    prediction = model.predict(input_arr)  # Model Prediction
    result_index = np.argmax(prediction)  # Get Class Index
    return result_index

# --------------------------- Sidebar ---------------------------
st.sidebar.markdown(
    "<h2 style='text-align: left; color: #2E8B57;'>🌿 <b>GreenCure</b></h2>",
    unsafe_allow_html=True
)

st.sidebar.markdown("<h3 style='color: white;'>Dashboard</h3>", unsafe_allow_html=True)

# 🔹 Fix: Removed duplicate page selection that caused extra "main" and "Disease Recognition"
app_mode = st.sidebar.radio("Navigate to", ["Home", "About", "Disease Recognition"])

# --------------------------- Home Page ---------------------------
if app_mode == "Home":
    st.markdown("<h1 style='text-align: center; color: #2E8B57;'>🌿 GreenCure</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Empowering Farmers with AI-based Plant Disease Detection</h3>", unsafe_allow_html=True)

    # Enlarged Image and Centered Predict Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("home_page.jpeg", use_container_width=True)  # Enlarged Image
        if st.button("Predict Disease", use_container_width=True):
            st.switch_page("pages/Disease_recognition.py")  # Correct file name


# --------------------------- About Page ---------------------------
elif app_mode == "About":
    st.markdown("<h1 style='text-align: center;'>About GreenCure</h1>", unsafe_allow_html=True)
    st.write("""
    ### **Project Details**
    **GreenCure** is an AI-based plant disease detection system that helps farmers and gardeners identify plant diseases using image recognition technology.

    **Key Features:**
    - Deep learning model trained on a vast dataset of healthy and diseased plant images.
    - Instant disease identification with high accuracy.
    - User-friendly interface for easy image upload and analysis.

    **How It Works:**
    1. Upload a clear image of the affected plant.
    2. The AI model analyzes the image and detects any disease.
    3. Results are displayed instantly with suggestions.

    **Dataset Distribution:**
    - Training: 70,295 images
    - Validation: 17,572 images
    - Testing: 33 images

    *GreenCure aims to assist farmers in protecting crops and improving agricultural productivity!*
    """)

# --------------------------- Disease Recognition Page ---------------------------
elif app_mode == "Disease Recognition":
    st.markdown("<h1 style='text-align: center;'>Disease Recognition</h1>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        test_image = st.file_uploader("Upload a Plant Image", type=["jpg", "png", "jpeg"])

        if test_image is not None:
            img = Image.open(test_image)
            st.image(img, use_container_width=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Analyze", use_container_width=True):
            with st.spinner("Analyzing... Please wait..."):
                result_index = model_prediction(test_image)

                class_name = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
                    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
                    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy'
                ]

                st.success(f"✅ Prediction: **{class_name[result_index]}**")
