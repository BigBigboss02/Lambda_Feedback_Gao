import random 
base_training_arguments = {
    "num_train_epochs": 10,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "save_steps": 50,
    "logging_steps": 10,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "warmup_ratio": 0.1
}

# Function to generate a new hyperparameter set by modifying numeric values by -90% to +90%
def perturb_hyperparameters(base_hyperparams):
    new_hyperparams = base_hyperparams.copy()
    for key, value in base_hyperparams.items():
        if isinstance(value, (int, float)) and key not in ["gradient_accumulation_steps"]:  # Exclude non-tunable params
            factor = random.uniform(0.1, 1.9)  # -90% to +90%
            new_hyperparams[key] = max(1e-7, value * factor)  # Ensure positive values
    return new_hyperparams

params = perturb_hyperparameters(base_training_arguments)
print(params)

def generate_model_name(hyperparams):
    name_parts = [
        f"epochs_{int(hyperparams['num_train_epochs'])}",
        f"batch_{int(hyperparams['per_device_train_batch_size'])}",
        f"lr_{hyperparams['learning_rate']:.1e}",
        f"wd_{hyperparams['weight_decay']:.2e}",
        f"warmup_{hyperparams['warmup_ratio']:.2f}"
    ]
    return "trained_model_" + "_".join(name_parts)
import os

new_model = "/models"
new_model_path = os.path.join(new_model,generate_model_name(params))
print(new_model_path)