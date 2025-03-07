import h5py
import numpy as np
import matplotlib.pyplot as plt

file_path = r"D:\AP22110010458\BrainPulse\dataset\data\600.mat"

try:
    with h5py.File(file_path, "r") as mat_file:
        print("File loaded successfully!")
        print("Available datasets:", list(mat_file.keys()))

        # Extract `cjdata`
        cjdata = mat_file['cjdata']
        print("cjdata keys:", list(cjdata.keys()))

        # Extract MRI image
        image_data = np.array(cjdata['image'])

        # Extract tumor mask & border
        tumor_mask = np.array(cjdata['tumorMask'])
        tumor_border = np.array(cjdata['tumorBorder'])

        # Plot MRI, Tumor Mask, and Tumor Border
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image_data, cmap='gray')
        axes[0].set_title("MRI Scan")
        axes[0].axis("off")

        axes[1].imshow(tumor_mask, cmap='hot', alpha=0.6)
        axes[1].set_title("Tumor Mask")
        axes[1].axis("off")

        axes[2].imshow(image_data, cmap='gray')
        axes[2].imshow(tumor_border, cmap='jet', alpha=0.6)
        axes[2].set_title("Tumor Border")
        axes[2].axis("off")

        plt.show()

except FileNotFoundError:
    print(f"Error: The file '{file_path}' does not exist. Check the path.")
except Exception as e:
    print(f"Error loading file: {e}")
