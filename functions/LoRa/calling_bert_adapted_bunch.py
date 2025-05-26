import pandas as pd
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoTokenizer,BertModel

from adapters import AutoAdapterModel, AdapterConfig

# --- Load dataset ---
csv_path = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/functions/LoRa/data/minimum_logic_testing_dataset.csv"
df = pd.read_csv(csv_path)
result_path = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/test_results/one_to_one_tests/Base_controlled_variable_tests/BeRT_large/adapter.csv"
adapter_path = '/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/functions/LoRa/tuned_BeRT_adaptors/large_trained_adaptor/1805_example_logic_oldarg'
# --- Load tokenizer and base model ---
model_path = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/base_models/BeRT/bert-large-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_path)



device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
base_model = BertModel.from_pretrained(
    model_path, 
    torch_dtype=torch.float16, 
)
# Load adapter model
model = PeftModel.from_pretrained(base_model, adapter_path).to(device)
model.eval()


# --- Get BERT embedding ---
def get_word_embedding(word):
    inputs = tokenizer(word, return_tensors="pt", add_special_tokens=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 1, :].squeeze(0)

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