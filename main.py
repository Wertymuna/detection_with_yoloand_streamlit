import streamlit as st
from PIL import Image
import torch
import cv2
import numpy as np


st.markdown("""
    <style>
    .small-text {
        font-size: 50px;
    }
    </style>
""", unsafe_allow_html=True)


# Load YOLOv5 model dari PyTorch Hub
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

model = load_model()

# Setup halaman
st.set_page_config(page_title="Deteksi Objek dari Foto", layout="centered")
st.title("📸 Deteksi Objek dari Foto")
st.markdown("### 1️⃣ Upload foto berisi objek seperti orang, mobil, bola, atau gedung.")

# Upload foto
uploaded_file = st.file_uploader("Unggah gambar (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Baca gambar
    image = Image.open(uploaded_file).convert("RGB")

    # Langkah 1: Tampilkan gambar awal
    st.markdown("### 2️⃣ Gambar Asli")
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    # Tombol untuk mulai deteksi
    if st.button("🔍 Deteksi Objek"):
        with st.spinner("Mendeteksi objek..."):
            results = model(image)
            df = results.pandas().xyxy[0]

            # Konversi PIL ke OpenCV
            img_cv = np.array(image)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

            for i in range(len(df)):
                xmin, ymin, xmax, ymax = map(int, [df.iloc[i]['xmin'], df.iloc[i]['ymin'], df.iloc[i]['xmax'], df.iloc[i]['ymax']])
                label = df.iloc[i]['name']
                confidence = df.iloc[i]['confidence']

                # Gambar bounding box
                cv2.rectangle(img_cv, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

                # Tulis label dengan font kecil
                text = f"{label}"
                cv2.putText(img_cv, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Konversi kembali ke RGB untuk Streamlit
            img_result = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

            # Tampilkan gambar hasil deteksi
            st.markdown("### 3️⃣ Hasil Deteksi Objek")
            st.image(img_result, caption="Gambar dengan Deteksi (Font kecil)", use_column_width=True)

            # Tampilkan daftar objek
            labels = df['name'].value_counts()
            if not labels.empty:
                st.markdown("### 🏷️ Objek yang Ditemukan:")
                for label, count in labels.items():
                    st.markdown(f"<span style='font-size:14px'>• {label}: {count}x</span>", unsafe_allow_html=True)
            else:
                st.info("Tidak ada objek terdeteksi.")
