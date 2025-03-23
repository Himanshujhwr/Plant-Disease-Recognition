import streamlit as st

# Page Title
st.title("Disease Recognition")

# Upload Image
uploaded_file = st.file_uploader("Upload an image of the plant leaf", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
