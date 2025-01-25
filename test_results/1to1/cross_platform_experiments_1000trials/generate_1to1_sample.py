import pandas as pd
import random

# Function to generate semantic comparison examples
def generate_semantic_comparisons(n):
    examples = []
    true_pairs = [
        ("happy", "joyful"),
        ("science", "knowledge"),
        ("love", "affection"),
        ("big", "large"),
        ("quick", "fast"),
        ("slow", "leisurely"),
        ("hot", "Hot"),
        ("cold", "COLD"),
        ("color", "colour"),
        ('labor', 'labour'),
        ('walking','walkin'),
        ('go','goes')
    ]
    false_pairs = [
        ("cat", "dog"),
        ("bank", "actor"),
        ("red", "blue"),
        ("table", "cloud"),
        ("stone", "water"),
        ("happy", "sad"),
        ("science", "art"),
        ("technology", "music"),
        ("city", "forest"),
        ("movie", "book"),
        ("computer", "food")
    ]
    unsure_pairs = [
        ("science", "physics"),
        ("movie", "filmography"),
        ("art", "creativity"),
        ("technology", "programming"),
        ("city", "metropolis")
    ]

    for _ in range(n):
        category = random.choice(["true", "false", "unsure"])
        if category == "true":
            pair = random.choice(true_pairs)
            examples.append((*pair, "True"))
        elif category == "false":
            pair = random.choice(false_pairs)
            examples.append((*pair, "False"))
        else:  # category == "unsure"
            pair = random.choice(unsure_pairs)
            examples.append((*pair, "Unsure"))
    
    return examples

# Generate 1000 rows
semantic_comparisons = generate_semantic_comparisons(1000)

# Convert to DataFrame
df = pd.DataFrame(semantic_comparisons, columns=["Word1", "Word2", "Ground Truth"])

# Save as CSV
file_path = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/test_results/1to1/semantic_comparisons.csv"
df.to_csv(file_path, index=False)

file_path
