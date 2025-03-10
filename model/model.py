import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_custom_cnn(input_shape=(128, 128, 3), num_classes=3):
    l2_reg = regularizers.l2(1e-4)  # L2 Regularization (1e-4)

    model = models.Sequential()

    # Conv Block 1
    model.add(layers.Conv2D(32, (3,3), padding='same', kernel_regularizer=l2_reg, input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))  

    # Conv Block 2
    model.add(layers.Conv2D(64, (3,3), padding='same', kernel_regularizer=l2_reg))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.3))  

    # Conv Block 3
    model.add(layers.Conv2D(128, (3,3), padding='same', kernel_regularizer=l2_reg))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.35))  

    # Conv Block 4
    model.add(layers.Conv2D(256, (3,3), padding='same', kernel_regularizer=l2_reg))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.4))  

    # Conv Block 5
    model.add(layers.Conv2D(512, (3,3), padding='same', kernel_regularizer=l2_reg))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.45))  

    # ✅ Global Average Pooling
    model.add(layers.GlobalAveragePooling2D())  

    # Fully Connected Layers
    model.add(layers.Dense(512, kernel_regularizer=l2_reg))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))  

    model.add(layers.Dense(256, kernel_regularizer=l2_reg))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.4))  

    # Output Layer (Softmax for multi-class classification)
    model.add(layers.Dense(num_classes, activation='softmax'))  

    return model

# Example usage
model = build_custom_cnn(input_shape=(128, 128, 3), num_classes=3)
model.summary()
