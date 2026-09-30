import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

img_size =(224, 224)

model = load_model("best_custom_cnn.h5", compile=False)

class_names =["glioma", "meningioma", "notumor", "pituitary"]

st.title("Brain tumour classification")
st.write("upload a brain MRI image")

uploaded_file = st.file_uploader("choose an image",type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image, caption="uploaded image", width=400)

    resized = cv2.resize(image, img_size)
    normalized = resized/255.0
    input_img = np.expand_dims(normalized, axis=0)

    predictions = model.predict(input_img)
    confidence = np.max(predictions)
    predicted_class = class_names[np.argmax(predictions)]

    st.title("Prediction result")
    st.write("tumor type:", predicted_class)
    st.write("confidence:", round(confidence * 100, 2), "%")