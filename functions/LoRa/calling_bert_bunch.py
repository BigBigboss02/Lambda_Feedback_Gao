import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F

# --- Load dataset ---
csv_path = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/functions/LoRa/data/minimum_logic_testing_dataset.csv"  # Update to your local path
df = pd.read_csv(csv_path)
result_path = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/test_results/one_to_one_tests/Base_controlled_variable_tests/BeRT_large/vanilla.csv"  # Update to your local path
# --- Load model and tokenizer ---
model_path = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/base_models/BeRT/bert-large-uncased"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)
model.eval()

# --- Get BERT embedding ---
def get_word_embedding(word):
    inputs = tokenizer(word, return_tensors="pt", add_special_tokens=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 1, :].squeeze(0)  # Use token at position 1

# --- Compute similarity ---
def compute_similarity(word1, word2, threshold=0.7):
    emb1 = get_word_embedding(word1)
    emb2 = get_word_embedding(word2)
    similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    decision = "true" if similarity > threshold else "false"
    return similarity, decision

# --- Process rows ---
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

# --- Append results ---
df["model_output_score"] = similarities
df["Model Output"] = decisions

# --- Save result ---
df.to_csv(result_path, index=False)
print("Saved results to", result_path)