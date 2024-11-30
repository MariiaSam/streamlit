import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps  

@st.cache_resource
def load_models():
    model1 = tf.keras.models.load_model("model_name.h5")
    model2 = tf.keras.models.load_model("model_vgg16.h5")
    return model1, model2

model1, model2 = load_models()

def preprocess_image(image):
    gray_image = ImageOps.grayscale(image)  
    resized_image = gray_image.resize((28, 28))  
    normalized_image = np.array(resized_image) / 255.0  
    input_image = np.expand_dims(normalized_image, axis=(0, -1))  
    return input_image

st.title("Класифікація зображень з використанням двох моделей")

model_option = st.selectbox(
    "Оберіть модель для класифікації",
    ("Модель 1", "Модель 2")
)

selected_model = model1 if model_option == "Модель 1" else model2

uploaded_file = st.file_uploader("Завантажте зображення для класифікації", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Завантажене зображення", use_column_width=True)

    input_image = preprocess_image(image)

    predictions = selected_model.predict(input_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    st.write(f"Обрана модель: {model_option}")
    st.write(f"Передбачений клас: {predicted_class}")
    st.write(f"Ймовірність: {confidence:.2f}")

    st.bar_chart(predictions.flatten())
