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
        label = int(mat_data['cjdata']['label'][0][0]) - 1  # Convert 1-indexed to 0-indexed

    # Normalize image
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Reduce brightness
    image = cv2.convertScaleAbs(image, alpha=0.85, beta=5)

    # Apply same transformations to both image and tumor mask
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    tumor_mask = cv2.rotate(tumor_mask, cv2.ROTATE_90_CLOCKWISE)

    # Convert grayscale to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Resize image and tumor mask together
    image_rgb = cv2.resize(image_rgb, IMG_SIZE)
    tumor_mask = cv2.resize(tumor_mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

    # Apply red color to tumor mask
    tumor_overlay = np.zeros_like(image_rgb)
    tumor_overlay[:, :, 0] = (tumor_mask * 255).astype(np.uint8)  # Apply red to tumor mask

    # Blend the tumor mask with the original image
    blended_image = cv2.addWeighted(image_rgb, 0.7, tumor_overlay, 0.3, 0)

    return blended_image, label

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
def get_tf_dataset(batch_size=32, split=0.8):
    images, labels = load_dataset()
    labels = tf.keras.utils.to_categorical(labels, num_classes=3)  # Adjust num_classes if needed

    # Splitting into train and validation sets
    split_idx = int(len(images) * split)
    train_images, val_images = images[:split_idx], images[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))

    # Shuffle, batch, and prefetch
    train_dataset = train_dataset.shuffle(len(train_images)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset

# Test if dataset is loading correctly
if __name__ == "__main__":
    train_dataset, val_dataset = get_tf_dataset(batch_size=1)  # Unpack both datasets
    sample_image, sample_label = next(iter(train_dataset))  # Use train_dataset

    # Convert to NumPy array for visualization
    sample_image = sample_image.numpy()[0]
    sample_label = np.argmax(sample_label.numpy()[0])  # Convert one-hot encoding to label index
    
    # Display image
    plt.imshow(sample_image)
    plt.title(f"Label: {sample_label}")
    plt.axis("off")
    plt.show()
