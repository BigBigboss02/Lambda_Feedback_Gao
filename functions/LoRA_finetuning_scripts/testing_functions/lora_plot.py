import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
file_path = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/test_results/1to1/finetuned_model_test/output_results_newparam.csv"
# Reload the CSV, skipping the first row which contains incorrect data
df = pd.read_csv(file_path, skiprows=1, names=["Word1", "Word2", "Ground Truth", "Model Output"])

# Convert 'Ground Truth' and 'Model Output' to boolean mappings
label_mapping = {"True": 1, "False": 0}
df["Ground Truth"] = df["Ground Truth"].astype(str).str.strip().str.capitalize().map(label_mapping)
df["Model Output"] = df["Model Output"].astype(str).str.strip().str.capitalize().map(label_mapping)

# Drop any NaN values that may have appeared due to mapping issues
filtered_data = df[['Ground Truth', 'Model Output']].dropna().astype(int)

# Compute the confusion matrix
conf_matrix = confusion_matrix(filtered_data['Ground Truth'], filtered_data['Model Output'], labels=[1, 0])

# Plot the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['True', 'False'],
            yticklabels=['True', 'False'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()
plt.show()

# Display the computed confusion matrix values
conf_matrix