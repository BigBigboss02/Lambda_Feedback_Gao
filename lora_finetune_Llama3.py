from transformers import LlamaTokenizer, LlamaForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import os

# Set the model and tokenizer paths
model_path = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Llama-3.2-1B"
tokenizer_path = model_path

# Load tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
model = LlamaForCausalLM.from_pretrained(model_path)

# Define the dataset (replace with your dataset)
# Example: load a Hugging Face dataset
data = load_dataset("wikitext", "wikitext-2-raw-v1")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = data.map(tokenize_function, batched=True, remove_columns=["text"])

# Split dataset into train and validation
train_dataset = tokenized_datasets["train"]
validation_dataset = tokenized_datasets["validation"]

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",          # output directory
    evaluation_strategy="epoch",    # evaluate each epoch
    learning_rate=5e-5,              # learning rate
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    num_train_epochs=3,              # total number of training epochs
    weight_decay=0.01,               # weight decay
    save_strategy="epoch",          # save checkpoint after each epoch
    save_total_limit=2,              # limit the total amount of checkpoints
    logging_dir="./logs",           # directory for storing logs
    logging_steps=10,                # log every 10 steps
    push_to_hub=False                # set to True if you want to push the model to Hugging Face Hub
)

# Define Trainer
trainer = Trainer(
    model=model,                         # the pre-trained model
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # training dataset
    eval_dataset=validation_dataset,     # evaluation dataset
    tokenizer=tokenizer                  # tokenizer
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./fine_tuned_llama")
tokenizer.save_pretrained("./fine_tuned_llama")

print("Fine-tuning completed!")
