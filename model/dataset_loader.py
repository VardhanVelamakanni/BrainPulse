import numpy as np
import h5py
import cv2
import os
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

# Define constants
IMG_SIZE = (128, 128)
DATASET_PATH = r"D:\AP22110010458\BrainPulse\dataset\data"
NUM_CLASSES = 3  # Adjust this if needed

# Function to load and preprocess a single .mat file
def load_mat_file(mat_path):
    with h5py.File(mat_path, 'r') as mat_data:
        image = np.array(mat_data['cjdata']['image'][()])
        tumor_mask = np.array(mat_data['cjdata']['tumorMask'][()])
        label = int(mat_data['cjdata']['label'][0][0]) - 1  # Convert 1-indexed to 0-indexed

    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    image = cv2.convertScaleAbs(image, alpha=0.85, beta=5)

    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    tumor_mask = cv2.rotate(tumor_mask, cv2.ROTATE_90_CLOCKWISE)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    image_rgb = cv2.resize(image_rgb, IMG_SIZE)
    tumor_mask = cv2.resize(tumor_mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

    tumor_overlay = np.zeros_like(image_rgb)
    tumor_overlay[:, :, 0] = (tumor_mask * 255).astype(np.uint8)

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

# Function to apply random zoom
def zoom_image(image, zoom_range=(0.9, 1.1)):
    """Apply random zoom by cropping and resizing."""
    zoom_factor = tf.random.uniform([], zoom_range[0], zoom_range[1])
    img_shape = tf.shape(image)[:2]  # Get image height and width
    new_height = tf.cast(zoom_factor * tf.cast(img_shape[0], tf.float32), tf.int32)
    new_width = tf.cast(zoom_factor * tf.cast(img_shape[1], tf.float32), tf.int32)

    image = tf.image.resize_with_crop_or_pad(image, new_height, new_width)
    image = tf.image.resize(image, IMG_SIZE)

    return image

# Data Augmentation Function
def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    angles = tf.random.uniform([], -20, 20) * np.pi / 180
    image = tfa.image.rotate(image, angles)
    image = zoom_image(image, (0.9, 1.1))  # Apply zoom effect
    
    return image, label

# Convert dataset to TensorFlow dataset
def get_tf_dataset(batch_size=32, split=0.8):
    images, labels = load_dataset()
    labels = tf.keras.utils.to_categorical(labels, num_classes=NUM_CLASSES)

    split_idx = int(len(images) * split)
    train_images, val_images = images[:split_idx], images[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))

    train_dataset = train_dataset.shuffle(len(train_images)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Compute class counts for better dataset analysis
    class_counts = np.bincount(np.argmax(labels, axis=1))

    return train_dataset, val_dataset, class_counts  # Now it returns 3 values


if __name__ == "__main__":
    BATCH_SIZE = 32
    train_dataset, val_dataset = get_tf_dataset(batch_size=BATCH_SIZE)
    sample_image, sample_label = next(iter(train_dataset))
    sample_image = sample_image.numpy()[0]
    sample_label = np.argmax(sample_label.numpy()[0])
    plt.imshow(sample_image)
    plt.title(f"Label: {sample_label}")
    plt.axis("off")
    plt.show()
