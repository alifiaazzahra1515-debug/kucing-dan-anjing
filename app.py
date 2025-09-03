import streamlit as st
import tensorflow as tf
import numpy as np
import requests
from PIL import Image
import os

# ========================
# CONFIGURASI APLIKASI
# ========================
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱🐶",
    layout="centered"
)

# URL model di Hugging Face (SavedModel .keras)
MODEL_URL = "https://huggingface.co/alifia1/catvsdog/resolve/main/model_mobilenetv2.keras"
MODEL_PATH = "model_mobilenetv2.keras"

# ========================
# LOAD MODEL (cache)
# ========================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("🔽 Mengunduh model dari Hugging Face..."):
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ========================
# ANTARMUKA
# ========================
st.markdown(
    """
    <h1 style='text-align: center; color: #ff6600;'>
        🐶🐱 Cat vs Dog Classifier
    </h1>
    <p style='text-align: center;'>
        Unggah gambar kucing atau anjing dan biarkan AI menebak! <br>
        Dibangun dengan TensorFlow & Streamlit 🚀
    </p>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("📤 Upload gambar (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)

with col1:
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Gambar yang diupload", use_column_width=True)

with col2:
    if uploaded_file:
        with st.spinner("🔍 Menganalisis gambar..."):
            # Preprocessing
            img = image.resize((150, 150))  # sesuaikan dengan input model
            img_array = tf.keras.utils.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Prediksi
            prediction = model.predict(img_array, verbose=0)[0][0]
            confidence = prediction if prediction > 0.5 else 1 - prediction
            label = "🐶 Dog" if prediction > 0.5 else "🐱 Cat"

        # Hasil prediksi
        st.success(f"Hasil Prediksi: **{label}**")
        st.progress(float(confidence))
        st.write(f"Confidence: **{confidence:.2%}**")

# ========================
# PETUNJUK
# ========================
st.markdown("---")
st.subheader("📘 Petunjuk Penggunaan")
st.write(
    """
    1. Klik tombol **Upload** untuk memilih gambar kucing atau anjing.
    2. Tunggu beberapa detik hingga model selesai menganalisis.
    3. Lihat hasil prediksi beserta tingkat confidence.  
    """
)

if st.button("🔄 Reset Aplikasi"):
    st.cache_resource.clear()
    st.experimental_rerun()
