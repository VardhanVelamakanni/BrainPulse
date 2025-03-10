import tensorflow as tf
from tensorflow.keras import layers, models, applications, regularizers

def build_pretrained_cnn(input_shape=(128, 128, 3), num_classes=3, trainable=False):
    base_model = applications.EfficientNetB0(
        weights='imagenet',  # Use pretrained ImageNet weights
        include_top=False,   # Remove fully connected layers
        input_shape=input_shape
    )

    base_model.trainable = trainable  # Freeze or fine-tune the model

    model = models.Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())  # Feature aggregation

    # Fully Connected Layers with L2 Regularization and Dropout
    model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))  # L2 Regularization
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))  # Dropout for regularization

    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))  # L2 Regularization
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    # Output Layer (Softmax for multi-class classification)
    model.add(layers.Dense(num_classes, activation='softmax'))  

    return model

# Learning Rate Scheduler
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4, decay_steps=10000, decay_rate=0.9
)

# Compile the model
model = build_pretrained_cnn(input_shape=(128, 128, 3), num_classes=3, trainable=False)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
