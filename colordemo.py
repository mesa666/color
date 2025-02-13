import os
import glob
import json
import uuid
import numpy as np
import cv2
import colorsys
import streamlit as st
from git import Repo

# ===================================================
# Generate Unique User ID and Setup Persistence File
# ===================================================
if "user_id" not in st.session_state:
    st.session_state["user_id"] = str(uuid.uuid4())
user_id = st.session_state["user_id"]
PERSISTENCE_FILE = f"slider_values_{user_id}.json"

# ===================================================
# GitPython Persistence Functions
# ===================================================
def load_slider_values():
    """
    Pulls the latest changes from the remote repository and then loads
    the persistence JSON file for the current user.
    """
    try:
        repo = Repo(os.getcwd())
        origin = repo.remote(name='origin')
        origin.pull()  # Pull latest changes from GitHub
    except Exception as e:
        st.error("Error pulling from GitHub: " + str(e))
    if os.path.exists(PERSISTENCE_FILE):
        with open(PERSISTENCE_FILE, "r") as f:
            return json.load(f)
    else:
        return {}

def save_slider_values(slider_values):
    """
    Saves the slider values to the local JSON file and then commits and pushes
    the file to the GitHub repository.
    
    IMPORTANT:
      - For this to work in Streamlit Cloud, you must have set up your remote URL
        to include your GitHub personal access token (e.g., via st.secrets) or ensure
        that your repository is configured to allow push access.
    """
    with open(PERSISTENCE_FILE, "w") as f:
        json.dump(slider_values, f)

# ===================================================
# Global Constants and Preset Stain References
# ===================================================
Io = 255          # Transmitted light intensity
alpha = 1         # Tolerance for pseudo-min and pseudo-max
beta = 0.15       # OD threshold for transparent pixels

# Preset (original) reference stain vectors (H&E)
HERef_init = np.array([[0.5626, 0.2159],
                       [0.7201, 0.8012],
                       [0.4062, 0.5581]])
maxCRef = np.array([2, 1])

# Precompute the original HSV values from HERef_init.
rgb_h_orig = HERef_init[:, 0] / np.linalg.norm(HERef_init[:, 0])
rgb_e_orig = HERef_init[:, 1] / np.linalg.norm(HERef_init[:, 1])
h_h_orig, s_h_orig, v_h_orig = colorsys.rgb_to_hsv(*rgb_h_orig)
h_e_orig, s_e_orig, v_e_orig = colorsys.rgb_to_hsv(*rgb_e_orig)
# Scale HSV values to slider ranges.
h_h_orig, s_h_orig, v_h_orig = h_h_orig * 360, s_h_orig * 100, v_h_orig * 100
h_e_orig, s_e_orig, v_e_orig = h_e_orig * 360, s_e_orig * 100, v_e_orig * 100

# ===================================================
# Inject Custom CSS for Colored Sliders
# ===================================================
st.markdown(
    """
    <style>
    /* Hue sliders: for Hematoxylin and Eosin */
    .hue-slider input[type=range]::-webkit-slider-runnable-track {
        background: linear-gradient(to right, red, yellow, lime, cyan, blue, magenta, red);
        height: 8px;
        border-radius: 5px;
    }
    .hue-slider input[type=range]::-moz-range-track {
        background: linear-gradient(to right, red, yellow, lime, cyan, blue, magenta, red);
        height: 8px;
        border-radius: 5px;
    }
    
    /* Saturation sliders */
    .saturation-slider input[type=range]::-webkit-slider-runnable-track {
        background: linear-gradient(to right, lightblue, blue);
        height: 8px;
        border-radius: 5px;
    }
    .saturation-slider input[type=range]::-moz-range-track {
        background: linear-gradient(to right, lightblue, blue);
        height: 8px;
        border-radius: 5px;
    }
    
    /* Value sliders */
    .value-slider input[type=range]::-webkit-slider-runnable-track {
        background: linear-gradient(to right, darkorange, orange, yellow);
        height: 8px;
        border-radius: 5px;
    }
    .value-slider input[type=range]::-moz-range-track {
        background: linear-gradient(to right, darkorange, orange, yellow);
        height: 8px;
        border-radius: 5px;
    }
    
    /* Common slider thumb styling */
    input[type=range]::-webkit-slider-thumb {
        -webkit-appearance: none;
        border: 1px solid #000;
        height: 20px;
        width: 20px;
        border-radius: 50%;
        background: #fff;
        margin-top: -6px;
        cursor: pointer;
    }
    input[type=range]::-moz-range-thumb {
        border: 1px solid #000;
        height: 20px;
        width: 20px;
        border-radius: 50%;
        background: #fff;
        cursor: pointer;
    }
    </style>
    """, unsafe_allow_html=True
)

# ===================================================
# Functions for Image Processing and Caching
# ===================================================
@st.cache_data(show_spinner=False)
def load_and_precompute(image_path):
    """
    Loads an image, converts it to RGB, and precomputes
    the stain deconvolution parameters.
    
    Returns a dictionary with:
      - 'img': the RGB image
      - 'h', 'w', 'c': image dimensions
      - 'C2': the full concentration matrix (scaled)
      - 'C2_row0', 'C2_row1': the rows corresponding to the two stains.
    """
    img = cv2.imread(image_path, 1)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape
    img_reshaped = img.reshape((-1, 3))
    
    # Compute optical density (OD)
    OD = -np.log10(np.maximum((img_reshaped.astype(np.float64) + 1) / Io, 1e-6))
    ODhat = OD[~np.any(OD < beta, axis=1)]
    
    # Eigen-decomposition to estimate stain vectors.
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    That = ODhat.dot(eigvecs[:, 1:3])
    phi = np.arctan2(That[:, 1], That[:, 0])
    
    # Determine extreme angles.
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)
    
    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
    
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        HE = np.array((vMax[:, 0], vMin[:, 0])).T
    
    # Solve for concentration matrix: OD = HE * C.
    Y = OD.T
    C, _, _, _ = np.linalg.lstsq(HE + 1e-6, Y, rcond=None)
    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    tmp = np.divide(maxC, maxCRef)
    C2 = np.divide(C, tmp[:, np.newaxis])
    
    return {
        'img': img,
        'h': h,
        'w': w,
        'c': c,
        'C2': C2,
        'C2_row0': C2[0, :],
        'C2_row1': C2[1, :]
    }

def compute_stain_images(data, use_original, slider_values):
    """
    Based on the loaded image data and the current stain color settings,
    computes and returns the three images:
      - Normalized H&E
      - H (Hematoxylin) Component
      - E (Eosin) Component
    """
    h_val, w_val = data['h'], data['w']
    C2_row0 = data['C2_row0']
    C2_row1 = data['C2_row1']
    
    # Choose which HSV values to use.
    if use_original:
        h_h, s_h, v_h = h_h_orig, s_h_orig, v_h_orig
        h_e, s_e, v_e = h_e_orig, s_e_orig, v_e_orig
    else:
        h_h = slider_values["H_H"]
        s_h = slider_values["S_H"]
        v_h = slider_values["V_H"]
        h_e = slider_values["H_E"]
        s_e = slider_values["S_E"]
        v_e = slider_values["V_E"]
    
    # Convert HSV to RGB.
    rgb_h = colorsys.hsv_to_rgb(h_h / 360.0, s_h / 100.0, v_h / 100.0)
    rgb_e = colorsys.hsv_to_rgb(h_e / 360.0, s_e / 100.0, v_e / 100.0)
    HE_h = np.array([rgb_h[0], rgb_h[1], rgb_h[2]])
    HE_e = np.array([rgb_e[0], rgb_e[1], rgb_e[2]])
    
    # Compute H component image.
    H_factor = np.outer(-HE_h, C2_row0)
    H_update = Io * np.exp(H_factor)
    H_update = np.clip(H_update, 0, 255)
    H_update = H_update.T.reshape((h_val, w_val, 3)).astype(np.uint8)
    
    # Compute E component image.
    E_factor = np.outer(-HE_e, C2_row1)
    E_update = Io * np.exp(E_factor)
    E_update = np.clip(E_update, 0, 255)
    E_update = E_update.T.reshape((h_val, w_val, 3)).astype(np.uint8)
    
    # Combine the two stain components (here a simple average).
    Inorm_update = 0.5 * H_update + 0.5 * E_update

    # Normalize the combined image to span the full 0-255 range.
    min_val = Inorm_update.min()
    max_val = Inorm_update.max()
    if max_val > min_val:
        Inorm_update = (Inorm_update - min_val) / (max_val - min_val) * 255
    else:
        Inorm_update = np.zeros_like(Inorm_update)
    Inorm_update = Inorm_update.astype(np.uint8)
    
    return Inorm_update, H_update, E_update

# ===================================================
# Main Streamlit Application
# ===================================================
def main():
    st.title("H&E Stain Normalization")
    
    # --- Setup: Image Files ---
    folder_path = r"img"  # Adjust as needed.
    file_pattern = os.path.join(folder_path, "*.png")
    file_list = sorted(glob.glob(file_pattern))
    
    if not file_list:
        st.error("No images found in the specified folder!")
        return
    
    # Initialize current image index in session_state.
    if "current_file_index" not in st.session_state:
        st.session_state.current_file_index = 4 if len(file_list) > 4 else 0

    # --- Navigation Controls ---
    col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
    with col_nav1:
        if st.button("Previous"):
            if st.session_state.current_file_index > 0:
                st.session_state.current_file_index -= 1
    with col_nav3:
        if st.button("Next"):
            if st.session_state.current_file_index < len(file_list) - 1:
                st.session_state.current_file_index += 1

    current_file = file_list[st.session_state.current_file_index]
    st.write(f"**File {st.session_state.current_file_index+1} of {len(file_list)}:** {os.path.basename(current_file)}")
    
    # --- Load and Precompute Image Data ---
    data = load_and_precompute(current_file)
    
    # --- Color Mode and Slider Controls ---
    st.sidebar.header("Stain Color Settings")
    color_mode = st.sidebar.radio("Color Mode", options=["Original Colors", "Slider Colors"])
    use_original = (color_mode == "Original Colors")
    
    # Load persistent slider values for this user.
    saved_values = load_slider_values()
    default_H_H = saved_values.get("H_H", h_h_orig)
    default_S_H = saved_values.get("S_H", s_h_orig)
    default_V_H = saved_values.get("V_H", v_h_orig)
    default_H_E = saved_values.get("H_E", h_e_orig)
    default_S_E = saved_values.get("S_E", s_e_orig)
    default_V_E = saved_values.get("V_E", v_e_orig)
    
    # Create (and preserve) slider values using unique keys.
    slider_values = {}
    if not use_original:
        with st.sidebar.container():
            st.markdown('<div class="hue-slider">', unsafe_allow_html=True)
            slider_values["H_H"] = st.slider("H_H (Hue for Hematoxylin)", 0, 360, int(default_H_H), key="H_H_val")
            st.markdown('</div>', unsafe_allow_html=True)
        with st.sidebar.container():
            st.markdown('<div class="saturation-slider">', unsafe_allow_html=True)
            slider_values["S_H"] = st.slider("S_H (Saturation for Hematoxylin)", 0, 100, int(default_S_H), key="S_H_val")
            st.markdown('</div>', unsafe_allow_html=True)
        with st.sidebar.container():
            st.markdown('<div class="value-slider">', unsafe_allow_html=True)
            slider_values["V_H"] = st.slider("V_H (Value for Hematoxylin)", 0, 100, int(default_V_H), key="V_H_val")
            st.markdown('</div>', unsafe_allow_html=True)
        with st.sidebar.container():
            st.markdown('<div class="hue-slider">', unsafe_allow_html=True)
            slider_values["H_E"] = st.slider("H_E (Hue for Eosin)", 0, 360, int(default_H_E), key="H_E_val")
            st.markdown('</div>', unsafe_allow_html=True)
        with st.sidebar.container():
            st.markdown('<div class="saturation-slider">', unsafe_allow_html=True)
            slider_values["S_E"] = st.slider("S_E (Saturation for Eosin)", 0, 100, int(default_S_E), key="S_E_val")
            st.markdown('</div>', unsafe_allow_html=True)
        with st.sidebar.container():
            st.markdown('<div class="value-slider">', unsafe_allow_html=True)
            slider_values["V_E"] = st.slider("V_E (Value for Eosin)", 0, 100, int(default_V_E), key="V_E_val")
            st.markdown('</div>', unsafe_allow_html=True)
        
        new_values = {
            "H_H": st.session_state["H_H_val"],
            "S_H": st.session_state["S_H_val"],
            "V_H": st.session_state["V_H_val"],
            "H_E": st.session_state["H_E_val"],
            "S_E": st.session_state["S_E_val"],
            "V_E": st.session_state["V_E_val"]
        }
        save_slider_values(new_values)
        st.sidebar.success(new_values)
    else:
        # When using original values, assign the preset values.
        slider_values["H_H"] = h_h_orig
        slider_values["S_H"] = s_h_orig
        slider_values["V_H"] = v_h_orig
        slider_values["H_E"] = h_e_orig
        slider_values["S_E"] = s_e_orig
        slider_values["V_E"] = v_e_orig

    # --- Compute the Stain-Processed Images ---
    Inorm_update, H_update, E_update = compute_stain_images(data, use_original, slider_values)
    
    # --- Display the Images ---
    st.subheader("Output Images")
    col1, col2, col3 = st.columns(3)
    col1.image(Inorm_update, caption="Normalized H&E", use_container_width=True)
    col2.image(H_update, caption="H Component", use_container_width=True)
    col3.image(E_update, caption="E Component", use_container_width=True)

if __name__ == '__main__':
    main()
