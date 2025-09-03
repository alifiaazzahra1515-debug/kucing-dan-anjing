import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from huggingface_hub import hf_hub_download

# ===== Load Model dari Hugging Face Hub =====
REPO_ID = "username/model-mobilenetv2"   # ganti dengan repo_id Anda
FILENAME = "model_mobilenetv2.keras"

@st.cache_resource
def load_hf_model():
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    model = load_model(model_path)
    return model

model = load_hf_model()
IMG_SIZE = (224, 224)  # ukuran input MobileNetV2

# ===== UI Streamlit =====
st.set_page_config(
    page_title="MobilenetV2 Image Classifier",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 MobileNetV2 Image Classifier")
st.markdown(
    """
    ### Upload gambar Anda untuk diklasifikasi  
    Model ini diambil langsung dari **Hugging Face Hub**.  
    """
)

# Upload file
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar yang diupload
    image_display = Image.open(uploaded_file).convert("RGB")
    st.image(image_display, caption="Gambar yang diupload", use_column_width=True)

    # Preprocess gambar
    img = image_display.resize(IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0  # normalisasi

    # Prediksi
    preds = model.predict(x)
    pred_class = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))

    # ===== Output =====
    st.subheader("🔎 Hasil Prediksi")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Class", pred_class)
    with col2:
        st.metric("Confidence", f"{confidence*100:.2f}%")

    st.markdown("### 📊 Probabilitas per Kelas")
    st.json(preds.tolist())
