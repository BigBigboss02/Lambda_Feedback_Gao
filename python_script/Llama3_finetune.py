import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

# Set the path to your model and the CSV file
model_path = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Llama-3.2-1B"
csv_path = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Lambda_Feedback_Gao\docs\wsn_finetuning_material.csv"  
output_dir = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\fine_tuned_vs1"

# Load the CSV as a dataset
df = pd.read_csv(csv_path)
dataset = Dataset.from_pandas(df)

# Prepare the data for the model
def preprocess_data(examples):
    return {"input_ids": tokenizer(examples["Question"], truncation=True, padding="max_length", max_length=128)["input_ids"],
            "labels": tokenizer(examples["Answer"], truncation=True, padding="max_length", max_length=128)["input_ids"]}

# Load tokenizer and model
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
# Set the pad_token to the eos_token if not already defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Preprocess the data
dataset = dataset.map(preprocess_data)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.gradient_checkpointing_enable()  # Enable gradient checkpointing
torch.cuda.empty_cache()  # Clear GPU cache


# Split the dataset into train and eval datasets
train_dataset = dataset.shuffle(seed=42).select(range(80))  # 80% training
eval_dataset = dataset.shuffle(seed=42).select(range(80, 100))  # 20% evaluation

# Update TrainingArguments
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,
    #per_device_train_batch_size=4,
    per_device_train_batch_size=1,  # Reduce batch size
    gradient_accumulation_steps=8,
    save_steps=500,
    save_total_limit=2,
    evaluation_strategy="steps",  # Keep evaluation
    eval_steps=100,
    logging_dir=os.path.join(output_dir, "logs"),
    logging_steps=10,
    learning_rate=5e-5,
    fp16=True,  # Enable mixed precision if using GPU
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Add eval_dataset here
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model fine-tuned and saved to {output_dir}")
