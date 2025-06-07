from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F

# Load the model and tokenizer
model_path = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/base_models/BeRT/bert-large-uncased"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)

# Function to get BERT embeddings for a single word
def get_word_embedding(word):
    inputs = tokenizer(word, return_tensors="pt", add_special_tokens=True)
    outputs = model(**inputs)
    # Extract the embedding of the [CLS] token or average the last hidden states
    return outputs.last_hidden_state[:, 1, :].squeeze(0)  # Use token at position 1 (word itself)

# Compare two words
def are_words_similar(word1, word2, threshold=0.7):
    emb1 = get_word_embedding(word1)
    emb2 = get_word_embedding(word2)
    
    # Compute cosine similarity
    similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
    
    # Return true if similarity is above threshold
    return "true" if similarity.item() > threshold else "false"

# Test words
print(are_words_similar("blue", "cyan"))