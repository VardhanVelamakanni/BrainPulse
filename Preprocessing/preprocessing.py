import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt

# Load the .mat file using h5py
mat_path = r"D:\AP22110010458\BrainPulse\dataset\data\600.mat"
with h5py.File(mat_path, 'r') as mat_data:
    image = np.array(mat_data['cjdata']['image'])
    tumor_mask = np.array(mat_data['cjdata']['tumorMask'])

# Normalize image for better contrast
image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Reduce brightness slightly
image = cv2.convertScaleAbs(image, alpha=0.85, beta=5)

# Rotate 90° to the right
image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
tumor_mask = cv2.rotate(tumor_mask, cv2.ROTATE_90_CLOCKWISE)

# **Convert grayscale to RGB**
image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

# **Apply red highlight with transparency**
overlay = image_rgb.copy()
overlay[tumor_mask > 0] = [255, 0, 0]  # Red color

# **Blend the tumor highlight with better visibility**
blended = cv2.addWeighted(image_rgb, 0.7, overlay, 0.3, 0)

# **Display the processed MRI**
plt.figure(figsize=(6, 6))
plt.imshow(blended)
plt.axis("off")
plt.title("MRI with 90° Right Rotation & Tumor Highlight")
plt.show()
