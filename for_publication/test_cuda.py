import subprocess
import importlib.util  # ✅ import util explicitly

def check_command(cmd):
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True, text=True)
        return output.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e.output.strip()}"

def check_module(name):
    spec = importlib.util.find_spec(name)
    return spec is not None

print("="*60)
print("🔍 Checking CUDA installation (via nvidia-smi):")
print(check_command("nvidia-smi"))
print("="*60)

# Check PyTorch
if check_module("torch"):
    import torch
    print("✅ PyTorch is installed")
    print("   Version:", torch.__version__)
    print("   CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("   GPU count:", torch.cuda.device_count())
        print("   Current device:", torch.cuda.get_device_name(0))
else:
    print("❌ PyTorch not installed")

print("="*60)

# Check TensorFlow
if check_module("tensorflow"):
    import tensorflow as tf
    print("✅ TensorFlow is installed")
    print("   Version:", tf.__version__)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("   GPUs available:", [gpu.name for gpu in gpus])
    else:
        print("   No GPU detected by TensorFlow")
else:
    print("❌ TensorFlow not installed")

print("="*60)
