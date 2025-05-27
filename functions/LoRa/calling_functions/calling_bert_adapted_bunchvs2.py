import pandas as pd
import torch
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig
from transformers import BertTokenizer, BertModel

# --- Paths ---
csv_path = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/functions/LoRa/data/minimum_logic_testing_dataset.csv"
result_path = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/test_results/one_to_one_tests/Base_controlled_variable_tests/BeRT_large/adapter.csv"
adapter_path = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/functions/LoRa/tuned_BeRT_adaptors/large_trained_adaptor/1805_example_logic_oldarg"
model_path = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/base_models/BeRT/bert-large-uncased"

# --- Load dataset ---
df = pd.read_csv(csv_path)

# --- Device setup ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- Load tokenizer and base model ---
tokenizer = BertTokenizer.from_pretrained(model_path)
base_model = BertModel.from_pretrained(model_path).to(device)

# --- LoRA config for loading ---
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query", "key", "value", "dense"],
    lora_dropout=0.1,
    bias="none",
    task_type="FEATURE_EXTRACTION",
)

# --- Wrap and load LoRA adapter ---
model = get_peft_model(base_model, lora_config)
model.load_adapter(adapter_path, adapter_name="default", is_trainable=False)
model.to(device)
model.eval()

# --- Function to get BERT embedding ---
def get_word_embedding(word):
    inputs = tokenizer(word, return_tensors="pt", add_special_tokens=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 1, :].squeeze(0)

# --- Compute cosine similarity ---
def compute_similarity(word1, word2, threshold=0.7):
    emb1 = get_word_embedding(word1)
    emb2 = get_word_embedding(word2)
    similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    decision = "true" if similarity > threshold else "false"
    return similarity, decision

# --- Process dataset ---
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

# --- Add and save results ---
df["model_output_score"] = similarities
df["Model Output"] = decisions
df.to_csv(result_path, index=False)
print("Saved results to", result_path)