import os
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from peft import get_peft_model, LoraConfig

# --- Paths ---
csv_path = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/functions/LoRa/data/minimum_logic_testing_dataset.csv"
adapter_root = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/functions/LoRa/adaptors/tuned_BeRT_adaptors/base_trained_adaptor"
model_path = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/base_models/BeRT/bert-large-uncased"
output_base = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/test_results/one_to_one_tests/LoRa_controlled_variable_tests/BeRT_large"

# --- Load dataset ---
df = pd.read_csv(csv_path)

# --- Load tokenizer and base model ---
tokenizer = BertTokenizer.from_pretrained(model_path)
base_model = BertModel.from_pretrained(model_path)

# --- Define dummy LoraConfig ---
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query", "key", "value", "dense"],
    lora_dropout=0.1,
    bias="none",
    task_type="FEATURE_EXTRACTION",
)

# --- List adapters ---
adapters = [name for name in os.listdir(adapter_root) if os.path.isdir(os.path.join(adapter_root, name))]

# --- Loop over adapters ---
for adapter_name in adapters:
    print(f"Processing adapter: {adapter_name}")
    
    # Reload base model and apply LoRA
    model = BertModel.from_pretrained(model_path)
    model = get_peft_model(model, lora_config)
    adapter_path = os.path.join(adapter_root, adapter_name)
    model.load_adapter(adapter_path, adapter_name="default", is_trainable=False)
    model.eval()

    # Define embedding function
    def get_word_embedding(word):
        inputs = tokenizer(word, return_tensors="pt", add_special_tokens=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 1, :].squeeze(0)

    # Define similarity function
    def compute_similarity(word1, word2, threshold=0.7):
        emb1 = get_word_embedding(word1)
        emb2 = get_word_embedding(word2)
        similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        decision = "true" if similarity > threshold else "false"
        return similarity, decision

    # Process each row
    similarities, decisions = [], []
    for text in df["text"]:
        try:
            word1 = text.split("Word1:")[1].split("Word2:")[0].strip()
            word2 = text.split("Word2:")[1].strip()
            sim, dec = compute_similarity(word1, word2)
            similarities.append(sim)
            decisions.append(dec)
        except Exception as e:
            similarities.append(None)
            decisions.append("error")

    # Save result for current adapter
    df[f"{adapter_name}_score"] = similarities
    df[f"{adapter_name}_output"] = decisions
    output_csv = os.path.join(output_base, f"{adapter_name}.csv")
    df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")