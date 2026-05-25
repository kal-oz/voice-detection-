import tensorflow as tf

# ----------------------------
# Load Dataset
# ----------------------------
DATA_DIR = "spectrograms"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# ----------------------------
# Class Names
# ----------------------------
class_names = train_ds.class_names
print("Classes:", class_names)

# ----------------------------
# Normalize Data
# ----------------------------
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(
    lambda x, y: (normalization_layer(x), y)
)

val_ds = val_ds.map(
    lambda x, y: (normalization_layer(x), y)
)

# ----------------------------
# Optimize Performance
# ----------------------------
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(
    buffer_size=AUTOTUNE
)

val_ds = val_ds.cache().prefetch(
    buffer_size=AUTOTUNE
)

# ----------------------------
# Data Augmentation
# ----------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# ----------------------------
# Build CNN Model
# ----------------------------
model = tf.keras.Sequential([

    # Input layer FIRST
    tf.keras.layers.Input(shape=(128, 128, 3)),

    # Data augmentation
    data_augmentation,

    # CNN layers
    tf.keras.layers.Conv2D(
        16,
        (3, 3),
        activation='relu'
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation='relu'
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation='relu'
    ),
    tf.keras.layers.MaxPooling2D(),

    # Flatten
    tf.keras.layers.Flatten(),

    # Dense layer
    tf.keras.layers.Dense(
        128,
        activation='relu'
    ),

    # Dropout
    tf.keras.layers.Dropout(0.5),

    # Output layer
    tf.keras.layers.Dense(
        len(class_names),
        activation='softmax'
    )
])

# ----------------------------
# Compile Model
# ----------------------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ----------------------------
# Early Stopping
# ----------------------------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

# ----------------------------
# Train Model
# ----------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=[early_stop]
)

# ----------------------------
# Evaluate Model
# ----------------------------
loss, accuracy = model.evaluate(val_ds)

print("Validation Loss:", loss)
print("Validation Accuracy:", accuracy)

# ----------------------------
# Save Model
# ----------------------------
model.save("audio_classifier_model.h5")

print("✅ Model training complete!")
