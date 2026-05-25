import tensorflow as tf

IMG_SIZE = (128, 128)
NUM_CLASSES = 4  # dog, drilling, engine, siren

model = tf.keras.Sequential([
    
    # Input layer
    tf.keras.layers.Input(shape=(128, 128, 3)),

    # Convolution block 1
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    # Convolution block 2
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    # Convolution block 3
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    # Flatten
    tf.keras.layers.Flatten(),

    # Dense layers
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    # Output layer
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()
