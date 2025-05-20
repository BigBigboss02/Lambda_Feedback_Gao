from transformers import LlamaTokenizer, LlamaForCausalLM, Trainer, TrainingArguments

from datasets import load_dataset, Dataset
import json
import sentencepiece
import os

from transformers import LlamaTokenizer, LlamaForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import json
import sentencepiece
import os

# Set the model and tokenizer paths
model_path = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Llama-3.2-1B"
tokenizer_path = model_path

# Load tokenizer and model
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

# Ensure the tokenizer has a pad_token
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))  # Resize embeddings to accommodate new token


# tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
model = LlamaForCausalLM.from_pretrained(model_path)

# Load your dataset
data_path = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Llama-3.2-1B\promts\examples1.json"
with open(data_path, "r") as file:
    raw_data = json.load(file)

# Convert to Hugging Face Dataset
dataset = Dataset.from_list(raw_data)

# Tokenize the dataset
def tokenize_function(examples):
    inputs = tokenizer(examples["original"], padding="max_length", truncation=True, max_length=512)
    # Use input_ids as labels (shifted during training for causal language modeling)
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs



tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["original"])

# Split dataset into train and validation sets
split_datasets = tokenized_datasets.train_test_split(test_size=0.2)
train_dataset = split_datasets["train"]
validation_dataset = split_datasets["test"]

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
trainer.save_model(r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Lambda_Feedback_Gao\LoRa\lora_results\vs1\fine_tuned_llama")
tokenizer.save_pretrained(r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Lambda_Feedback_Gao\LoRa\lora_results\vs1\fine_tuned_llama")

print("Fine-tuning completed!")
