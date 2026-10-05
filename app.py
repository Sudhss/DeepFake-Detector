import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from torchvision import transforms
from PIL import Image
from huggingface_hub import hf_hub_download

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Synthetic Media Forensics",
    layout="centered"
)

# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 5px;
        color: #E2E8F0;
    }
    .subtitle {
        text-align: center;
        font-size: 16px;
        color: #94A3B8;
        margin-bottom: 35px;
        font-family: monospace;
    }
    .prediction-real {
        text-align: center;
        font-size: 28px;
        font-weight: 700;
        margin-top: 20px;
        color: #22C55E;
    }
    .prediction-fake {
        text-align: center;
        font-size: 28px;
        font-weight: 700;
        margin-top: 20px;
        color: #EF4444;
    }
    .confidence {
        text-align: center;
        font-size: 18px;
        color: #CBD5E1;
        margin-top: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="main-title">Synthetic Media Forensics</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">EfficientNet-B0 Convolutional Analysis Engine</div>', unsafe_allow_html=True)

# ============================================================
# DEVICE & MODEL CONFIGURATION
# ============================================================

device = torch.device("cpu")
MODEL_REPO = "RohitSharma69/deepfake-efficientnet-b0"
MODEL_FILENAME = "best_efficientnet_b0.pth"

# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner="Initializing neural network...")
def load_model():
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model

model = load_model()

# ============================================================
# IMAGE PREPROCESSING
# ============================================================

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================
# UPLOADER & INFERENCE
# ============================================================

uploaded_file = st.file_uploader("Select media file for analysis (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Source Media", use_container_width=True)

    with col2:
        if st.button("Execute Analysis", use_container_width=True):
            with st.spinner("Processing tensor data..."):
                image_tensor = eval_transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = torch.softmax(outputs, dim=1)[0]
                
                real_probability = probabilities[0].item()
                fake_probability = probabilities[1].item()
                predicted_class = torch.argmax(probabilities).item()

                st.divider()

                if predicted_class == 0:
                    st.markdown(f'<div class="prediction-real">VERDICT: AUTHENTIC</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="confidence">Confidence Score: {real_probability * 100:.2f}%</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-fake">VERDICT: SYNTHETIC</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="confidence">Confidence Score: {fake_probability * 100:.2f}%</div>', unsafe_allow_html=True)
                
                st.write("")
                st.progress(fake_probability, text=f"Synthetic probability threshold: {fake_probability * 100:.2f}%")

else:
    st.info("System Ready. Awaiting media input.")

# ============================================================
# SYSTEM DIAGNOSTICS
# ============================================================

with st.expander("System Architecture Details"):
    st.write("""
    **Core Architecture:** EfficientNet-B0 Backbone  
    **Input Tensor:** 224x224 RGB  
    **Normalization:** ImageNet Standards  
    **Output Head:** Binary Classification (Authentic vs Synthetic)  
    
    *Note: Model is optimized for detecting StyleGAN/StyleGAN2 structural artifacts. Modern diffusion models (e.g., Midjourney v6) may exhibit different frequency domain characteristics.*
    """)
