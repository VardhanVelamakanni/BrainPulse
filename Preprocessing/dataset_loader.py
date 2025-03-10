import numpy as np
import h5py
import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt

# Define image size
IMG_SIZE = (224, 224)

# Path to dataset folder containing .mat files
DATASET_PATH = r"D:\AP22110010458\BrainPulse\dataset\data"

# Function to load and preprocess a single .mat file
def load_mat_file(mat_path):
    with h5py.File(mat_path, 'r') as mat_data:
        image = np.array(mat_data['cjdata']['image'])
        tumor_mask = np.array(mat_data['cjdata']['tumorMask'])
        label = int(mat_data['cjdata']['label'][0][0]) - 1  # Convert MATLAB 1-indexed to 0-indexed
    
    # Normalize image
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Reduce brightness
    image = cv2.convertScaleAbs(image, alpha=0.85, beta=5)
    
    # Rotate 90° right
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    
    # Convert grayscale to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Resize to model input size
    image_rgb = cv2.resize(image_rgb, IMG_SIZE)
    
    return image_rgb, label

# Load all .mat files and create dataset
def load_dataset():
    images, labels = [], []
    for file_name in os.listdir(DATASET_PATH):
        if file_name.endswith(".mat"):
            img, lbl = load_mat_file(os.path.join(DATASET_PATH, file_name))
            images.append(img)
            labels.append(lbl)
    
    images = np.array(images) / 255.0  # Normalize to range [0,1]
    labels = np.array(labels)
    
    return images, labels

# Convert dataset to TensorFlow dataset
def get_tf_dataset(batch_size=32):
    images, labels = load_dataset()
    labels = tf.keras.utils.to_categorical(labels, num_classes=4)  # Assuming 4 tumor classes
    
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(len(images)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset
# Test if dataset is loading correctly
if __name__ == "__main__":
    dataset = get_tf_dataset(batch_size=1)
    sample_image, sample_label = next(iter(dataset))
    
    plt.imshow(sample_image[1])  # Show first image
    plt.title(f"Label: {sample_label.numpy()[1]}")
    plt.axis("off")
    plt.show()
