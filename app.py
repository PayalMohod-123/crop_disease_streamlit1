import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

def load_labels(filename="labels.txt"):
    with open(filename, "r") as f:
        return [line.strip() for line in f.readlines()]

@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image):
    img = image.resize((224, 224))
    img = np.expand_dims(img, axis=0)
    img = np.array(img, dtype=np.float32) / 255.0
    return img

def predict(image, model, labels):
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    input_data = preprocess_image(image)
    model.set_tensor(input_details[0]['index'], input_data)
    model.invoke()
    output = model.get_tensor(output_details[0]['index'])[0]
    pred_idx = np.argmax(output)
    confidence = float(output[pred_idx])
    return labels[pred_idx], confidence

st.title("🌿 Crop Disease Detection")
st.write("Upload a leaf image to check if it's *healthy* or *diseased*.")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf", use_column_width=True)

    model = load_model()
    labels = load_labels()

    label, confidence = predict(image, model, labels)

    st.success(f"Prediction: *{label}* ({confidence*100:.2f}% confidence)")
