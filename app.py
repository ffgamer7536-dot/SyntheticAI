import streamlit as st
import os
import torch
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# Import from project source
from src.utils import load_config, COLOR_MAP
from src.dataset import get_val_transforms
from src.model import create_model

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Gaia | Off-Road Autonomy", 
    page_icon="🌍", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CACHE MODEL LOADING ---
@st.cache_resource
def load_model_and_config():
    config = load_config('config.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(
        arch=config['model']['architecture'],
        backbone=config['model']['backbone'],
        weights=None,
        in_channels=config['model']['in_channels'],
        num_classes=config['model']['num_classes']
    ).to(device)
    
    weights_path = "saved_model_weights/best.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        model.eval()
    
    return model, config, device

model, config, device = load_model_and_config()

# --- HEADER ---
st.title("🌍 Gaia: Terrain Intelligence")
st.subheader("Robust Semantic Segmentation for Off-Road Autonomy")
st.markdown("*Bridging Synthetic Learning with Real-World Terrain Intelligence*")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Performance Dashboard", "🔍 Live Inference", "❗ Failure Cases Analysis"])

# --- TAB 1: DASHBOARD ---
with tab1:
    st.markdown("### 📈 Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Test mIoU", "30.92%", "Peak Val: 58.01%")
    col2.metric("Train Loss", "0.3226", "-")
    col3.metric("Validation Loss", "0.2993", "-")
    
    st.markdown("---")
    
    colA, colB = st.columns([1, 1])
    
    with colA:
        st.markdown("### 🎯 Per-Class Performance")
        st.markdown("Intersection over Union across key biological and geological targets within the off-road landscape.")
        
        # Exact values from PDF
        data = {
            "Class Name": ["Sky", "Landscape", "Dry Grass", "Dry Bushes", "Trees", "Rocks", "Lush Bushes", "Ground Clutter", "Flowers", "Logs"],
            "Approx. IoU": ["0.9764", "0.6432", "0.4551", "0.3194", "0.3186", "0.0267", "0.0005", "0.0000", "0.0000", "0.0000"]
        }
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
    with colB:
        st.markdown("### 📉 Training & Validation Loss Curve")
        if os.path.exists("runs/loss_curve.png"):
            st.image("runs/loss_curve.png", caption="Model Convergence Architecture", use_container_width=True)
        else:
            st.warning("Loss curve not generated locally.")

# --- TAB 2: LIVE INFERENCE ---
with tab2:
    st.markdown("### 🧠 Live Model Inference")
    st.markdown("Upload any real-world or synthetic desert image to run the **DeepLabV3+** pipeline in the backend.")
    
    uploaded_file = st.file_uploader("Upload an Off-Road Image (JPG/PNG)", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        try:
            raw_image = Image.open(uploaded_file).convert("RGB")
            img_np = np.array(raw_image)
            
            st.info("🔄 Running inference in the backend...")
            
            # Use transforms
            transforms = get_val_transforms(config['dataset']['img_height'], config['dataset']['img_width'])
            transformed = transforms(image=img_np)
            input_tensor = transformed['image'].unsqueeze(0).to(device)
            
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(input_tensor)
                    preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
            
            # Map predictions to RGB
            colored_pred = COLOR_MAP[preds]
                
            pred_image = Image.fromarray(colored_pred)
            
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.image(raw_image, caption="Original Image", use_container_width=True)
            with col_img2:
                st.image(pred_image, caption="DeepLabV3+ Semantic Prediction", use_container_width=True)
                
            st.success("✅ Inference Complete!")
            
        except Exception as e:
            st.error(f"Error running inference: {e}")

# --- TAB 3: FAILURE CASES ---
with tab3:
    st.markdown("### ⚠️ Automated Failure Case Flagging")
    st.markdown("Visualizing our lowest overlapping predictions to inform downstream architecture improvements.")
    
    # Collect all failure cases
    failure_dir = "runs/failure_cases/"
    if os.path.exists(failure_dir):
        failure_images = [f for f in os.listdir(failure_dir) if f.endswith(".png")]
        if failure_images:
            for img_name in failure_images[:5]:  # Display up to 5
                st.image(os.path.join(failure_dir, img_name), caption=img_name, use_container_width=True)
        else:
            st.info("No failure cases found.")
    else:
        st.info("Failure directory not populated.")

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## ⚙️ Model Context")
    st.markdown("**Architecture:** DeepLabV3+ (ResNet50)")
    st.markdown("**Dataset Input:** 512x512 RGB")
    st.markdown("**Classes Captured:** 10 Distinct Ecosystem Targets")
    st.markdown("---")
    st.markdown("### Future Optimizations")
    st.markdown("- Focal Loss Integration")
    st.markdown("- Class Weight Balancing")
    st.markdown("- Attention-based SegFormer Upgrades")
