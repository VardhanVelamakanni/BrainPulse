import numpy as np
import h5py
import cv2
import os
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from tensorflow.keras.applications import EfficientNetV2L  # Use EfficientNetV2-L
from tensorflow.keras.applications.efficientnet import preprocess_input  # EfficientNet preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Data Augmentation

# Define constants
IMG_SIZE = (128, 128)  # Updated model input size
DATASET_PATH = r"D:\AP22110010458\BrainPulse\dataset\data"
NUM_CLASSES = 3  # Update as needed

# Define Data Augmentation Generator
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Function to load and preprocess a single .mat file
def load_mat_file(mat_path):
    with h5py.File(mat_path, 'r') as mat_data:
        image = np.array(mat_data['cjdata']['image'][()])
        tumor_mask = np.array(mat_data['cjdata']['tumorMask'][()])
        label = int(mat_data['cjdata']['label'][0][0]) - 1  # Convert 1-based index to 0-based

    # Normalize image intensity
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Apply contrast enhancement
    image = cv2.convertScaleAbs(image, alpha=0.85, beta=5)

    # Rotate images
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    tumor_mask = cv2.rotate(tumor_mask, cv2.ROTATE_90_CLOCKWISE)

    # Convert grayscale to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image_rgb = cv2.resize(image_rgb, IMG_SIZE)

    tumor_mask = cv2.resize(tumor_mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

    # Overlay tumor mask in red
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

    images = np.array(images, dtype=np.float32)  # Convert to float32 for model compatibility
    labels = np.array(labels)

    return images, labels

# Function to apply random zoom (fixed)
def zoom_image(image, zoom_range=(0.9, 1.1)):
    """Apply random zoom by cropping and resizing while ensuring valid dimensions."""
    zoom_factor = tf.random.uniform([], zoom_range[0], zoom_range[1])
    img_shape = tf.shape(image)[:2]  # Get image height and width

    new_height = tf.cast(zoom_factor * tf.cast(img_shape[0], tf.float32), tf.int32)
    new_width = tf.cast(zoom_factor * tf.cast(img_shape[1], tf.float32), tf.int32)

    # Ensure the crop size does not exceed original dimensions
    new_height = tf.minimum(new_height, img_shape[0])
    new_width = tf.minimum(new_width, img_shape[1])

    image = tf.image.resize_with_crop_or_pad(image, new_height, new_width)
    image = tf.image.resize(image, (128, 128))  # Match model input size

    return image

# Data Augmentation Function
def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

    # Apply small rotation
    angles = tf.random.uniform([], -20, 20) * np.pi / 180
    image = tfa.image.rotate(image, angles)

    # Apply zoom augmentation
    image = zoom_image(image, (0.9, 1.1))

    return image, label

# Convert dataset to TensorFlow dataset
def get_tf_dataset(batch_size=32, split=0.8):
    images, labels = load_dataset()
    
    # Convert labels to one-hot encoding
    labels = np.eye(NUM_CLASSES)[labels]

    # Train-test split
    split_idx = int(len(images) * split)
    train_images, val_images = images[:split_idx], images[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]

    # Normalize images using EfficientNet preprocessing
    train_images = preprocess_input(train_images)
    val_images = preprocess_input(val_images)

    # Apply data augmentation using ImageDataGenerator
    train_generator = datagen.flow(train_images, train_labels, batch_size=batch_size)

    # Convert to TensorFlow dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Compute class counts for dataset analysis
    class_counts = np.bincount(np.argmax(labels, axis=1))

    return train_generator, val_dataset, class_counts  # Returning class distribution

# Main execution block
if __name__ == "__main__":
    BATCH_SIZE = 32
    train_generator, val_dataset, class_counts = get_tf_dataset(batch_size=BATCH_SIZE)

    print("Class Distribution:", class_counts)

    # Display an augmented image sample
    sample_batch = next(iter(train_generator))
    sample_image = sample_batch[0][0]  # First image from the batch
    sample_label = np.argmax(sample_batch[1][0])  # First label

    plt.imshow(sample_image.astype(np.uint8))
    plt.title(f"Label: {sample_label}")
    plt.axis("off")
    plt.show()
