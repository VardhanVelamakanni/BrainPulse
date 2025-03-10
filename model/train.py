import tensorflow as tf
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np

# Import dataset loader function
from dataset_loader import get_tf_dataset  # Ensure this function correctly loads your dataset
from model import build_pretrained_cnn  # Import the pretrained CNN model

# 🔹 Hyperparameters
BATCH_SIZE = 32
EPOCHS = 25
FINE_TUNE_EPOCHS = 10  # Additional fine-tuning epochs
INITIAL_LEARNING_RATE = 1e-4  # ✅ Smaller initial LR for stability
FINE_TUNE_LEARNING_RATE = 1e-5  # ✅ Even lower LR for fine-tuning
INPUT_SHAPE = (128, 128, 3)
NUM_CLASSES = 3  # Ensure it matches dataset labels

# 🔹 Load dataset (Ensure it returns train & validation datasets)
train_dataset, val_dataset, class_counts = get_tf_dataset(batch_size=BATCH_SIZE)

# 🔹 Compute class weights (if dataset is imbalanced)
total_samples = np.sum(class_counts)
class_weights = {i: total_samples / (NUM_CLASSES * class_counts[i]) for i in range(NUM_CLASSES)}
print("Class Weights:", class_weights)

# 🔹 Build pretrained CNN model (EfficientNetB0 as feature extractor)
model = build_pretrained_cnn(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, trainable=False)

# 🔹 Compile model with AdamW optimizer
model.compile(
    optimizer=AdamW(learning_rate=INITIAL_LEARNING_RATE, weight_decay=1e-4),  # ✅ AdamW with weight decay
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# 🔹 Callbacks
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ModelCheckpoint("brain_tumor_pretrained_model.h5", save_best_only=True, save_weights_only=False, verbose=1)
]

# 🔹 Train model with class weights
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights  # ✅ Balance training if needed
)

# 🔹 Fine-tuning: Unfreeze last 10 layers & train with lower LR
for layer in model.layers[-10:]:  
    layer.trainable = True  # ✅ Unfreezing only last 10 layers

# ✅ Recompile after unfreezing
model.compile(
    optimizer=AdamW(learning_rate=FINE_TUNE_LEARNING_RATE, weight_decay=1e-4),  # ✅ Lower LR for fine-tuning
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ✅ Fine-tuning training
fine_tune_history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights
)

# 🔹 Plot accuracy & loss curves
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history.get("accuracy", []), label="Train Accuracy")
plt.plot(history.history.get("val_accuracy", []), label="Val Accuracy")
plt.plot(fine_tune_history.history.get("accuracy", []), label="Fine-Tune Train Accuracy", linestyle="dashed")
plt.plot(fine_tune_history.history.get("val_accuracy", []), label="Fine-Tune Val Accuracy", linestyle="dashed")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training & Validation Accuracy")

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history.get("loss", []), label="Train Loss")
plt.plot(history.history.get("val_loss", []), label="Val Loss")
plt.plot(fine_tune_history.history.get("loss", []), label="Fine-Tune Train Loss", linestyle="dashed")
plt.plot(fine_tune_history.history.get("val_loss", []), label="Fine-Tune Val Loss", linestyle="dashed")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training & Validation Loss")

plt.show()
