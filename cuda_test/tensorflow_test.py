import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# Check for GPU support
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Number of GPUs detected: {len(gpus)}")
    for gpu in gpus:
        print("GPU:", gpu.name)
else:
    print("No GPU detected. TensorFlow is using CPU.")
