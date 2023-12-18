import os
import io
import streamlit as st
import requests
from PIL import Image

host = os.getenv("MODEL_ADDRESS", default="0.0.0.0:9696")

lbl_mapping = { 0: "Bike", 1: "Car" }
inv_lbl_mapping = { v: k for k, v in lbl_mapping.items() }

# Streamlit app
st.title("Vehicle Prediction App")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # BytesIO used to create a bytes-like object from the uploaded file
    image_bytes = uploaded_file.read()
    pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # Showing the image
    st.image(pil_img, caption="Uploaded Image.", use_column_width=True)

    # Convert the image to jpg if png 
    if uploaded_file.type == "image/png":
        output = io.BytesIO()
        pil_img.save(output, format='JPEG')
        image_bytes = output.getvalue()

    # Send a request to the Flask app
    files = { 'file': ('image.jpg', image_bytes, 'image/jpg') }
    response = requests.post(f"http://{host}/predict", files=files)

    # Display prediction result
    if response.status_code == 200:
        result = response.json()
        print(result)
        class_prob = 1 - result['prob'] if result['pred_class_name'] == "Bike" else result['prob']
        st.success(f"Prediction: {result['pred_class_name']} (Probability: {class_prob:.4f})")
    else:
        st.error("Failed to make a prediction." + f"{response.status_code}")