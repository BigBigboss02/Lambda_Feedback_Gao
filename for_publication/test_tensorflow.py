import tensorflow as tf

# Check if GPU is detected
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow version:", tf.__version__)

# Simple training test
import numpy as np

# Dummy data
x = np.random.rand(1000, 10)
y = np.random.rand(1000, 1)

# Very small model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

print("\n>>> Training test (should use GPU if available)...")
model.fit(x, y, epochs=2, batch_size=32, verbose=1)

# Simple prediction test
print("\n>>> Prediction test...")
pred = model.predict(x[:5])
print("Sample predictions:", pred)