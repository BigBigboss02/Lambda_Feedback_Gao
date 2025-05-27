from transformers import BertTokenizer, BertModel
from peft import get_peft_model, LoraConfig
import torch
import torch.nn.functional as F

# --- Paths ---
model_path = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/base_models/BeRT/bert-large-uncased"
adapter_path = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/functions/LoRa/tuned_BeRT_adaptors/large_trained_adaptor/1805_example_logic_oldarg"

# --- Load tokenizer and base model ---
tokenizer = BertTokenizer.from_pretrained(model_path)
base_model = BertModel.from_pretrained(model_path)

# --- Define dummy LoraConfig for loading ---
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query", "key", "value", "dense"],
    lora_dropout=0.1,
    bias="none",
    task_type="FEATURE_EXTRACTION",  # Important for models like BertModel
)

# --- Wrap base model with LoRA ---
model = get_peft_model(base_model, lora_config)

# --- Load LoRA weights manually ---
model.load_adapter(adapter_path, adapter_name="default", is_trainable=False)

# --- Function to get embeddings ---
def get_word_embedding(word):
    inputs = tokenizer(word, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 1, :].squeeze(0)

# --- Compare ---
def are_words_similar(word1, word2, threshold=0.7):
    emb1 = get_word_embedding(word1)
    emb2 = get_word_embedding(word2)
    sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    return "true" if sim > threshold else "false"

# --- Test ---
print(are_words_similar("blue", "cyan"))