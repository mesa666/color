import os
import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
import colorsys

# ============================================
# Global Parameters and Settings
# ============================================
Io = 255          # Transmitted light intensity
alpha = 1         # Tolerance for pseudo-min and pseudo-max
beta = 0.15       # OD threshold for transparent pixels

# Preset (original) reference stain vectors (H&E)
HERef_init = np.array([[0.5626, 0.2159],
                       [0.7201, 0.8012],
                       [0.4062, 0.5581]])
                       
maxCRef = np.array([1.9705, 1.0308])

# Folder and file list (change folder and file type as needed)
folder_path = r"img"
file_pattern = os.path.join(folder_path, "*.png")
file_list = sorted(glob.glob(file_pattern))
current_file_index = 4

# Global variables for image-dependent precomputed values.
img_shape = None
h, w, c = None, None, None
C2 = None
C2_row0 = None
C2_row1 = None
current_img = None

# Global variable for the display axes (for the 3 subplots)
img_display = None

# Global flag for color mode: True = use original (preset) colors,
# False = use slider-defined colors.
use_original = False

# ----------------------------------------------
# Compute original HSV values from HERef_init.
# (These remain constant and are used when "Original Colors" is active.)
rgb_h_orig = HERef_init[:, 0] / np.linalg.norm(HERef_init[:, 0])
rgb_e_orig = HERef_init[:, 1] / np.linalg.norm(HERef_init[:, 1])
h_h_orig, s_h_orig, v_h_orig = colorsys.rgb_to_hsv(*rgb_h_orig)
h_e_orig, s_e_orig, v_e_orig = colorsys.rgb_to_hsv(*rgb_e_orig)
# Scale HSV values to typical slider ranges.
h_h_orig, s_h_orig, v_h_orig = h_h_orig * 360, s_h_orig * 100, v_h_orig * 100
h_e_orig, s_e_orig, v_e_orig = h_e_orig * 360, s_e_orig * 100, v_e_orig * 100

# ============================================
# Helper Functions
# ============================================
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
    print(f"maxC: {maxC}")
    tmp = np.divide(maxC, maxCRef)
    # tmp = np.divide(maxC, maxC)
    C2 = np.divide(C, tmp[:, np.newaxis])
    
    return {'img': img, 'h': h, 'w': w, 'c': c,
            'C2': C2, 'C2_row0': C2[0, :], 'C2_row1': C2[1, :]}

def add_slider_gradient(ax, vmin, vmax, cmap, orientation='horizontal', hsv_offset=False):
    """
    Adds a gradient background to a slider axis.
    
    Parameters:
      ax          : the axis to paint.
      vmin, vmax  : slider value limits.
      cmap        : colormap to use.
      orientation : 'horizontal' or 'vertical'.
      hsv_offset  : if True, offset hue values by 180°.
    """
    if orientation == 'horizontal':
        gradient = np.linspace(vmin, vmax, 256)
        if hsv_offset:
            gradient = (gradient + 180) % 360
        gradient = np.tile(gradient, (20, 1))
        extent = [vmin, vmax, 0, 1]
    else:
        gradient = np.linspace(vmin, vmax, 256)
        if hsv_offset:
            gradient = (gradient + 180) % 360
        gradient = np.tile(gradient, (20, 1)).T
        extent = [0, 1, vmin, vmax]
    
    ax.imshow(gradient, aspect='auto', cmap=cmap, extent=extent)
    ax.set_xticks([])
    ax.set_yticks([])

# ============================================
# Update Function for the Display
# ============================================
def update(val):
    """
    Updates the image display based on the current stain color settings.
    Uses the global flag `use_original` to decide whether to use the preset (original)
    stain colors or the slider-defined ones.
    """
    global h, w, C2_row0, C2_row1, use_original

    # Choose which HSV values to use.
    if use_original:
        # Use the original (preset) values.
        h_h, s_h, v_h = h_h_orig, s_h_orig, v_h_orig
        h_e, s_e, v_e = h_e_orig, s_e_orig, v_e_orig
    else:
        # Use the current slider values.
        h_h, s_h, v_h = slider_H_h.val, slider_H_s.val, slider_H_v.val
        h_e, s_e, v_e = slider_E_h.val, slider_E_s.val, slider_E_v.val

    # Convert HSV to RGB.
    rgb_h = colorsys.hsv_to_rgb(h_h / 360, s_h / 100, v_h / 100)
    rgb_e = colorsys.hsv_to_rgb(h_e / 360, s_e / 100, v_e / 100)
    
    # Form the local stain color vectors.
    HE_h = np.array([rgb_h[0], rgb_h[1], rgb_h[2]])
    HE_e = np.array([rgb_e[0], rgb_e[1], rgb_e[2]])
    
    # Use the precomputed concentration rows to recompute the stain components.
    H_factor = np.outer(-HE_h, C2_row0)
    H_update = Io * np.exp(H_factor)
    H_update[H_update > 255] = 254
    H_update = H_update.T.reshape((h, w, 3)).astype(np.uint8)
    
    E_factor = np.outer(-HE_e, C2_row1)
    E_update = Io * np.exp(E_factor)
    E_update[E_update > 255] = 254
    E_update = E_update.T.reshape((h, w, 3)).astype(np.uint8)
    
    # Combine the two stain components (here a simple average).
    Inorm_update = 0.5 * H_update + 0.5 * E_update
    Inorm_update = np.clip(Inorm_update, 0, 255).astype(np.uint8)
    
    # Update the three subplot displays.
    img_display[0].set_data(Inorm_update)
    img_display[1].set_data(H_update)
    img_display[2].set_data(E_update)
    plt.draw()

# ============================================
# Function to Load a New Image (by Index)
# ============================================
def load_new_image(index):
    """
    Loads the image at file_list[index], precomputes the necessary parameters,
    updates the global variables, and refreshes the display.
    """
    global current_img, h, w, c, C2, C2_row0, C2_row1, current_file_index
    file_path = file_list[index]
    print(f"Loading image: {file_path}")
    results = load_and_precompute(file_path)
    current_img = results['img']
    h, w, c = results['h'], results['w'], results['c']
    C2 = results['C2']
    C2_row0 = results['C2_row0']
    C2_row1 = results['C2_row1']
    
    fig.suptitle(f"File {index+1}/{len(file_list)}: {os.path.basename(file_path)}", fontsize=14)
    update(None)

# ============================================
# Figure, Sliders, and Buttons Setup
# ============================================
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# Create image displays for the normalized H&E, H component, and E component.
img_display = [axs[i].imshow(np.zeros((300, 300, 3), dtype=np.uint8)) for i in range(3)]
axs[0].set_title('Normalized H&E')
axs[1].set_title('H Component')
axs[2].set_title('E Component')
for ax in axs:
    ax.axis('off')
plt.subplots_adjust(left=0.1, bottom=0.45, top=0.85)

# ----- Create Slider Axes -----
ax_H_h = plt.axes([0.1, 0.33, 0.65, 0.03])
ax_H_s = plt.axes([0.1, 0.28, 0.65, 0.03])
ax_H_v = plt.axes([0.1, 0.23, 0.65, 0.03])
ax_E_h = plt.axes([0.1, 0.18, 0.65, 0.03])
ax_E_s = plt.axes([0.1, 0.13, 0.65, 0.03])
ax_E_v = plt.axes([0.1, 0.08, 0.65, 0.03])

# Add gradient backgrounds to slider axes.
add_slider_gradient(ax_H_h, 0, 360, cmap=plt.get_cmap('hsv'), hsv_offset=True)
add_slider_gradient(ax_E_h, 0, 360, cmap=plt.get_cmap('hsv'), hsv_offset=True)
add_slider_gradient(ax_H_s, 0, 100, cmap=plt.get_cmap('Blues'))
add_slider_gradient(ax_E_s, 0, 100, cmap=plt.get_cmap('Blues'))
add_slider_gradient(ax_H_v, 0, 100, cmap=plt.get_cmap('Oranges'))
add_slider_gradient(ax_E_v, 0, 100, cmap=plt.get_cmap('Oranges'))

# ----- Create Sliders (initial values based on the original reference)
# For initial slider values, use the original computed HSV from HERef_init.
slider_H_h = Slider(ax_H_h, 'H_H', 0, 360, valinit=h_h_orig)
slider_H_s = Slider(ax_H_s, 'S_H', 0, 100, valinit=s_h_orig)
slider_H_v = Slider(ax_H_v, 'V_H', 0, 100, valinit=v_h_orig)
slider_E_h = Slider(ax_E_h, 'H_E', 0, 360, valinit=h_e_orig)
slider_E_s = Slider(ax_E_s, 'S_E', 0, 100, valinit=s_e_orig)
slider_E_v = Slider(ax_E_v, 'V_E', 0, 100, valinit=v_e_orig)

# Connect slider events to the update function.
slider_H_h.on_changed(update)
slider_H_s.on_changed(update)
slider_H_v.on_changed(update)
slider_E_h.on_changed(update)
slider_E_s.on_changed(update)
slider_E_v.on_changed(update)

# ----- Create Navigation Buttons for Images -----
button_ax_prev = plt.axes([0.8, 0.15, 0.1, 0.05])
button_ax_next = plt.axes([0.8, 0.08, 0.1, 0.05])
button_prev = Button(button_ax_prev, 'Previous')
button_next = Button(button_ax_next, 'Next')

def prev_image(event):
    global current_file_index
    if current_file_index > 0:
        current_file_index -= 1
        load_new_image(current_file_index)
    else:
        print("Already at the first image.")

def next_image(event):
    global current_file_index
    if current_file_index < len(file_list) - 1:
        current_file_index += 1
        load_new_image(current_file_index)
    else:
        print("Already at the last image.")

button_prev.on_clicked(prev_image)
button_next.on_clicked(next_image)

# ----- Create Buttons to Toggle Color Modes -----
button_ax_original = plt.axes([0.8, 0.30, 0.1, 0.05])
button_ax_slider = plt.axes([0.8, 0.23, 0.1, 0.05])
button_original = Button(button_ax_original, 'Original Colors')
button_slider = Button(button_ax_slider, 'Slider Colors')

def set_original(event):
    global use_original
    use_original = True
    update(None)  # Refresh the display using original colors.

def set_slider(event):
    global use_original
    use_original = False
    update(None)  # Refresh the display using slider-defined colors.

button_original.on_clicked(set_original)
button_slider.on_clicked(set_slider)

# ============================================
# Initialize with the First Image
# ============================================
load_new_image(current_file_index)
plt.show()
