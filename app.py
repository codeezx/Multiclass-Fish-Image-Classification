import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

st.title("🐟 Multiclass Fish Image Classifier")
st.write("Upload a fish image and the model will predict its species.")

# Load the model
@st.cache_resource
def load_trained_model():
    return load_model("best_fish_model.h5")  

model = load_trained_model()

class_names = [
    'bass',
    'black_sea_sprat',
    'red_sea_bream',
    'fish',  
    'striped_red_mullet',
    'red_mullet',
    'hourse_mackerel',
    'shrimp',
    'trout',
    'gilt_head_bream',
    'sea_bass'
]

# Upload image
uploaded_file = st.file_uploader("Upload a fish image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image_pil = Image.open(uploaded_file).convert("RGB")
    st.image(image_pil, caption="Uploaded Image", use_column_width=True)
    

    # Preprocess the image
    img = image_pil.resize((224, 224))  # Resize as per model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    # Show result
    st.success(f"**Predicted Fish Species:** {predicted_class}")
