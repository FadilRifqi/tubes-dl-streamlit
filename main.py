import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import os


# -------------------------
# KONFIGURASI STREMLIT
# -------------------------
st.set_page_config(
    page_title="Sistem Presensi Mahasiswa - Face Recognition (ViT)",
    page_icon="🎓",
    layout="wide"
)

# -------------------------
# PATH MODEL & LABEL
# -------------------------
MODEL_PATH = os.path.join("model", "model.pth")
LABELS_PATH = os.path.join("model", "label.txt")


# -------------------------
# LOAD LABEL
# -------------------------
def load_labels(path):
    if not os.path.exists(path):
        st.error(f"❌ File label tidak ditemukan: {path}")
        return []
    with open(path, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


# -------------------------
# BUILD MODEL UNTUK STATE_DICT
# -------------------------
def build_vit_model(num_classes=70):
    model = models.vit_b_16(weights=None)

    # Sesuaikan HEAD sesuai training kamu
    model.heads.head = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.heads.head.in_features, num_classes)
    )
    return model


# -------------------------
# LOAD MODEL (STATE_DICT)
# -------------------------
@st.cache_resource
def load_model(path, num_classes=70):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(path):
        st.error(f"❌ File model tidak ditemukan: {path}")
        return None, None

    # Bangun ulang struktur ViT
    model = build_vit_model(num_classes)

    # Load bobot state_dict
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model, device


# -------------------------
# PREPROCESSING GAMBAR
# (sesuai training kamu)
# -------------------------
def process_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    return preprocess(image).unsqueeze(0)  # Tambah batch dim


# -------------------------
# UI APLIKASI
# -------------------------
def main():
    st.title("🎓 Sistem Presensi Mahasiswa")
    st.markdown("### Deep Learning - Face Recognition (Vision Transformer)")
    st.markdown("---")

    # Sidebar info
    st.sidebar.header("ℹ Informasi")
    st.sidebar.info("Aplikasi ini menggunakan model Vision Transformer (ViT) untuk mengenali wajah mahasiswa.")

    # Load resource
    class_names = load_labels(LABELS_PATH)
    model, device = load_model(MODEL_PATH, num_classes=len(class_names))

    if model is None or not class_names:
        st.warning("⚠ Pastikan file model (.pth) dan label (.txt) tersedia di folder 'model/'.")
        return

    col1, col2 = st.columns([1, 1])

    # -------------------------------------
    # Upload Foto
    # -------------------------------------
    with col1:
        st.subheader("1. Upload Foto Wajah")
        uploaded = st.file_uploader("Pilih file gambar...", type=["jpg", "jpeg", "png"])

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Foto yang diupload", use_column_width=True)

            if st.button("🔍 Identifikasi Wajah"):
                with st.spinner("Sedang memproses..."):
                    input_tensor = process_image(image).to(device)

                    with torch.no_grad():
                        output = model(input_tensor)
                        probabilities = F.softmax(output, dim=1)
                        conf, idx = torch.max(probabilities, 1)

                    confidence = conf.item() * 100
                    predicted_name = class_names[idx.item()]

                # -------------------------------------
                # Kolom Hasil
                # -------------------------------------
                with col2:
                    st.subheader("2. Hasil Identifikasi")

                    if confidence >= 60:
                        st.success(f"✔ Wajah teridentifikasi sebagai **{predicted_name}**")
                        st.metric("Confidence", f"{confidence:.2f}%")
                        st.progress(int(confidence))
                        st.info(f"Kehadiran telah dicatat untuk: **{predicted_name}**")

                    else:
                        st.error("❌ Wajah tidak dikenali / Confidence rendah")
                        st.write(f"Prediksi terdekat: {predicted_name} ({confidence:.2f}%)")


# -------------------------
# RUN APP
# -------------------------
if __name__ == "__main__":
    main()
