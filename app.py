import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the pre-trained model
model_path = "celebrity_classification_model.h5"  # Change this to your saved model path
loaded_model = load_model(model_path)

# Manually define class indices based on the order in which the classes were present during training
class_indices = {0: "celebrity_1", 1: "celebrity_2", 2: "celebrity_3", 3: "celebrity_4", 4: "celebrity_5"}

# Streamlit app
st.title("Celebrity Image Classification")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Preprocess the image
    img = image.load_img(uploaded_file, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    prediction = loaded_model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # Display the predicted celebrity
    st.subheader("Prediction:")
    st.write(f"This image is classified as {class_indices[predicted_class]}.")

    # Display the prediction probabilities
    st.subheader("Prediction Probabilities:")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_indices[i]}: {prob:.2%}")
