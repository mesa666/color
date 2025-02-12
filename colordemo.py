import streamlit as st
import cv2
import numpy as np
import glob
import os
import colorsys

# ============================================
# Global Parameters and Settings
# ============================================
Io = 255
alpha = 1
beta = 0.15

# Preset reference stain vectors
HERef_init = np.array([[0.5626, 0.2159],
                       [0.7201, 0.8012],
                       [0.4062, 0.5581]])
maxCRef = np.array([1.9705, 1.0308])

# Initialize session state
if 'file_index' not in st.session_state:
    st.session_state.file_index = 0
if 'use_original' not in st.session_state:
    st.session_state.use_original = False

# ============================================
# Helper Functions
# ============================================
def load_and_precompute(image_path):
    """Same as original"""
    img = cv2.imread(image_path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape
    img_reshaped = img.reshape((-1, 3))
    
    OD = -np.log10(np.maximum((img_reshaped.astype(np.float64) + 1) / Io, 1e-6))
    ODhat = OD[~np.any(OD < beta, axis=1)]
    
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    That = ODhat.dot(eigvecs[:, 1:3])
    phi = np.arctan2(That[:, 1], That[:, 0])
    
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)
    
    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
    
    HE = np.array((vMin[:, 0], vMax[:, 0])).T if vMin[0] > vMax[0] else np.array((vMax[:, 0], vMin[:, 0])).T
    
    Y = OD.T
    C, _, _, _ = np.linalg.lstsq(HE + 1e-6, Y, rcond=None)
    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    tmp = np.divide(maxC, maxCRef)
    C2 = np.divide(C, tmp[:, np.newaxis])
    
    return {
        'img': img,
        'h': h, 'w': w, 'c': c,
        'C2_row0': C2[0, :],
        'C2_row1': C2[1, :]
    }

# ============================================
# Streamlit UI
# ============================================
st.title("H&E Stain Separation Demo")

# File selection
folder_path = r"img"
file_list = sorted(glob.glob(os.path.join(folder_path, "*.png")))

# Navigation controls
col1, col2, col3 = st.columns([1,1,2])
with col1:
    if st.button("Previous") and st.session_state.file_index > 0:
        st.session_state.file_index -= 1
with col2:
    if st.button("Next") and st.session_state.file_index < len(file_list)-1:
        st.session_state.file_index += 1

# Load current image
current_file = file_list[st.session_state.file_index]
precomputed = load_and_precompute(current_file)

# Color controls
st.sidebar.header("Color Controls")
st.sidebar.checkbox("Use Original Colors", key='use_original')

if not st.session_state.use_original:
    # H Stain controls
    st.sidebar.subheader("H Stain")
    h_h = st.sidebar.slider("Hue (H)", 0, 360, 210)
    h_s = st.sidebar.slider("Saturation (H)", 0, 100, 80)
    h_v = st.sidebar.slider("Value (H)", 0, 100, 80)

    # E Stain controls
    st.sidebar.subheader("E Stain")
    e_h = st.sidebar.slider("Hue (E)", 0, 360, 50)
    e_s = st.sidebar.slider("Saturation (E)", 0, 100, 80)
    e_v = st.sidebar.slider("Value (E)", 0, 100, 80)
else:
    # Use original values
    rgb_h_orig = HERef_init[:, 0] / np.linalg.norm(HERef_init[:, 0])
    rgb_e_orig = HERef_init[:, 1] / np.linalg.norm(HERef_init[:, 1])
    h_h, h_s, h_v = [x*100 for x in colorsys.rgb_to_hsv(*rgb_h_orig)]
    e_h, e_s, e_v = [x*100 for x in colorsys.rgb_to_hsv(*rgb_e_orig)]

# ============================================
# Image Processing
# ============================================
def process_image(h_h, h_s, h_v, e_h, e_s, e_v, precomputed):
    """Process image with current parameters"""
    # Convert HSV to RGB
    rgb_h = colorsys.hsv_to_rgb(h_h/360, h_s/100, h_v/100)
    rgb_e = colorsys.hsv_to_rgb(e_h/360, e_s/100, e_v/100)
    
    # Reconstruct images
    H = (Io * np.exp(-np.outer(precomputed['C2_row0'], rgb_h))).reshape(precomputed['h'], precomputed['w'], 3)
    E = (Io * np.exp(-np.outer(precomputed['C2_row1'], rgb_e))).reshape(precomputed['h'], precomputed['w'], 3)
    Combined = np.clip(0.5*H + 0.5*E, 0, 255).astype(np.uint8)
    
    return Combined, H.astype(np.uint8), E.astype(np.uint8)
# Display images
combined, h_img, e_img = process_image(
    h_h, h_s, h_v,
    e_h, e_s, e_v,
    precomputed
)

col1, col2, col3 = st.columns(3)
with col1:
    st.image(combined, caption="Combined", use_column_width=True)
with col2:
    st.image(h_img, caption="H Component", use_column_width=True)
with col3:
    st.image(e_img, caption="E Component", use_column_width=True)

st.caption(f"Showing file {st.session_state.file_index+1}/{len(file_list)}: {os.path.basename(current_file)}")
