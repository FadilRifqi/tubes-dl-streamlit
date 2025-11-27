import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import os

# ----------------------------
# CUSTOM STYLE DASHBOARD KAMPUS
# ----------------------------
dashboard_css = """
<style>
    /* Background soft */
    body {
        background-color: #f7f5f6;
    }

    /* HEADER with custom palette */
    .header-container {
        background: linear-gradient(90deg, #c9989c 0%, #d5b4b6 20%, #dcc1c3 40%, #dcd1d1 55%, #cbc8ce 70%, #b2c1cd 80%, #9f9ebd 90%, #9592aa 100%);
        padding: 28px;
        border-radius: 12px;
        color: #2b1f20;
        margin-bottom: 18px;
    }
    .header-title {
        font-size: 34px;
        font-weight: 800;
        margin: 0;
        color: #2b1f20;
    }
    .header-subtitle {
        font-size: 16px;
        opacity: 0.95;
        color: #3a2a2b;
    }

    /* CARD */
    .card {
        background: white;
        padding: 22px;
        border-radius: 12px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.06);
        margin-bottom: 18px;
    }

    /* Camera box styling */
    .camera-box {
        display: flex;
        align-items: center;
        justify-content: center;
        border: 2px dashed #d5b4b6;
        padding: 16px;
        border-radius: 10px;
        background: linear-gradient(180deg, rgba(201,152,156,0.03), rgba(149,146,170,0.02));
    }

    /* Text */
    h2, h3, h4 {
        color: #3a2a2b;
        margin-bottom: 10px;
        font-weight: 700;
    }

    /* Result highlight */
    .result-ok {
        background: linear-gradient(90deg, rgba(178,193,205,0.12), rgba(159,158,189,0.06));
        padding: 14px;
        border-radius: 10px;
        color: #1f2b33;
        font-size: 18px;
        font-weight: 700;
    }
    .result-bad {
        background: linear-gradient(90deg, rgba(220,193,195,0.12), rgba(220,209,209,0.06));
        padding: 14px;
        border-radius: 10px;
        color: #5a1616;
        font-size: 18px;
        font-weight: 700;
    }

    /* small metric */
    .metric-small {
        font-size: 14px;
        color: #3a2a2b;
        margin-top: 8px;
    }

</style>
"""

st.markdown(dashboard_css, unsafe_allow_html=True)

# ----------------------------
# STREAMLIT CONFIG
# ----------------------------
st.set_page_config(
    page_title="Presensi Mahasiswa - Face Recognition",
    page_icon="🎓",
    layout="wide"
)

# ----------------------------
# MODEL & LABEL PATH
# ----------------------------
MODEL_PATH = "model/model.pth"
LABELS_PATH = "model/label.txt"

# ----------------------------
# LOAD LABEL
# ----------------------------
def load_labels(path):
    if not os.path.exists(path):
        st.error(f"❌ File label tidak ditemukan: {path}")
        return []
    return [line.strip() for line in open(path, "r").readlines()]

# ----------------------------
# BUILD MODEL
# ----------------------------
def build_vit_model(num_classes=70):
    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.heads.head.in_features, num_classes)
    )
    return model

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model(path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_vit_model(num_classes)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model, device

# ----------------------------
# PREPROCESS IMAGE
# ----------------------------
def process_image(image):
    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    return trans(image).unsqueeze(0)


# ----------------------------
# MAIN UI
# ----------------------------
def main():

    # HEADER DASHBOARD
    st.markdown("""
        <div class="header-container">
            <div class="header-title">🎓 Presensi Mahasiswa</div>
            <div class="header-subtitle">Face Recognition • Vision Transformer (ViT)</div>
        </div>
    """, unsafe_allow_html=True)

    # Load model & labels
    class_names = load_labels(LABELS_PATH)
    if len(class_names) == 0:
        return

    model, device = load_model(MODEL_PATH, len(class_names))

    # Camera card (full width, on top)
    camera = st.camera_input("Arahkan wajah ke kamera")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Prediction card (below camera)
    st.markdown("## 🔎 Hasil Identifikasi")

    if camera:
        img = Image.open(camera).convert("RGB")

        input_tensor = process_image(img).to(device)

        with st.spinner("Memproses wajah mahasiswa..."):
            with torch.no_grad():
                out = model(input_tensor)
                prob = F.softmax(out, 1)
                conf, idx = torch.max(prob, 1)

        pred_name = class_names[idx.item()]
        conf_val = conf.item() * 100

        if conf_val >= 60:
            st.markdown(f"""
                <div class='result-ok'>
                    ✔ Wajah dikenali sebagai {pred_name}
                </div>
            """, unsafe_allow_html=True)
            st.markdown(f"<div class='metric-small'>Confidence: {conf_val:.2f}%</div>", unsafe_allow_html=True)
            st.success(f"Kehadiran telah dicatat untuk: **{pred_name}**")
        else:
            st.markdown(f"""
                <div class='result-bad'>
                    ❌ Wajah tidak dikenali
                </div>
            """, unsafe_allow_html=True)
            st.write(f"Prediksi terdekat: {pred_name} ({conf_val:.2f}%)")

    else:
        st.info("Silakan ambil foto wajah terlebih dahulu.")

    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    main()
