import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

st.set_page_config(
    page_title="ACV Satellite Monitoring",
    page_icon="🛰️",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "artifacts" / "improved_cnn_best.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImprovedCNN(nn.Module):
    def __init__(self, in_channels=12, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Dropout(0.15),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Dropout(0.20),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


@st.cache_resource
def load_model():
    model = ImprovedCNN().to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


model = load_model()


def normalize_for_display(band):
    band = band.astype(np.float32)
    p2, p98 = np.percentile(band, (2, 98))

    if p98 - p2 < 1e-6:
        return np.zeros_like(band, dtype=np.float32)

    band = (band - p2) / (p98 - p2)
    return np.clip(band, 0, 1).astype(np.float32)


def make_rgb(image, r=3, g=2, b=1):
    rgb = np.stack([
        normalize_for_display(image[r]),
        normalize_for_display(image[g]),
        normalize_for_display(image[b]),
    ], axis=-1)
    return rgb


def preprocess_npz(uploaded_file, img_size=128):
    data = np.load(uploaded_file, allow_pickle=True)

    image = data["image"]
    label = data["label"]

    x = torch.tensor(image, dtype=torch.float32)

    x = F.interpolate(
        x.unsqueeze(0),
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False
    ).squeeze(0)

    x = (x - x.mean()) / (x.std() + 1e-6)
    x = x.unsqueeze(0).to(device)

    true_label = int((label > 0).any())

    return x, image, label, true_label


st.title("🛰️ ACV Satellite Monitoring")

st.markdown(
    """
Classificação de imagens de satélite para detecção de queimadas florestais utilizando CNNs treinadas com o dataset Sen2Fire.
"""
)

st.sidebar.header("Informações")
st.sidebar.write("Modelo final: ImprovedCNN")
st.sidebar.write("Dataset: Sen2Fire")
st.sidebar.write("Acurácia no teste: 0.83")
st.sidebar.write("F1 (fogo): 0.52")

uploaded_file = st.file_uploader(
    "Selecione um arquivo .npz do Sen2Fire",
    type=["npz"]
)

if uploaded_file is not None:
    x, image, label, true_label = preprocess_npz(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Composição RGB")
        st.image(make_rgb(image), clamp=True, channels="RGB")

    with col2:
        st.subheader("Label real")
        st.image((label > 0).astype(np.uint8) * 255, clamp=True, channels="L")

    with st.expander("Ver banda individual"):
        band_idx = st.slider(
            "Escolha uma banda",
            min_value=0,
            max_value=image.shape[0] - 1,
            value=0
        )
        st.image(
            normalize_for_display(image[band_idx]),
            clamp=True,
            channels="L",
            caption=f"Banda {band_idx}"
        )

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred = torch.argmax(probs).item()

    st.divider()
    st.subheader("Resultado")

    if pred == 1:
        st.error("🚨 COM FOGO")
    else:
        st.success("🌲 SEM FOGO")

    st.write(f"Confiança: {probs[pred].item():.2%}")
    st.write(f"Rótulo real: {'COM FOGO' if true_label else 'SEM FOGO'}")
    st.write(f"Probabilidade SEM FOGO: {probs[0].item():.2%}")
    st.write(f"Probabilidade COM FOGO: {probs[1].item():.2%}")