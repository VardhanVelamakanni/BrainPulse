import tensorflow as tf
from tensorflow.keras.optimizers import AdamW  
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np

# Import dataset loader function
from dataset_loader import get_tf_dataset  # Ensure this function exists and is correct
from model import build_custom_cnn  # Import the CNN model

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 5e-5  # ✅ Smaller initial LR
INPUT_SHAPE = (128, 128, 3)
NUM_CLASSES = 3  # Ensure it matches dataset labels

# ✅ Load dataset (Ensure it returns train & validation datasets)
train_dataset, val_dataset, class_counts = get_tf_dataset(batch_size=BATCH_SIZE)

# ✅ Compute class weights (if dataset is imbalanced)
total_samples = np.sum(class_counts)
class_weights = {i: total_samples / (NUM_CLASSES * class_counts[i]) for i in range(NUM_CLASSES)}
print("Class Weights:", class_weights)

# ✅ Build model
model = build_custom_cnn(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)

# ✅ Compile model with AdamW
model.compile(
    optimizer=AdamW(learning_rate=LEARNING_RATE, weight_decay=1e-4),  # ✅ Weight Decay in AdamW
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ✅ Callbacks
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),  # ✅ Adjusted patience
    ModelCheckpoint("brain_tumor_model.h5", save_best_only=True, save_weights_only=False, verbose=1)
]

# ✅ Train model with class weights
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights  # ✅ Balance training if needed
)

# ✅ Plot accuracy & loss curves
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history.get("accuracy", []), label="Train Accuracy")
plt.plot(history.history.get("val_accuracy", []), label="Val Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training & Validation Accuracy")

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history.get("loss", []), label="Train Loss")
plt.plot(history.history.get("val_loss", []), label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training & Validation Loss")

plt.show()
