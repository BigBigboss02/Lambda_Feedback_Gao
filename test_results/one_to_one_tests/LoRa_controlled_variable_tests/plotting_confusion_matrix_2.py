# Re-import necessary modules after code execution state reset
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Re-define paths
parsed_folder = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/test_results/one_to_one_tests/LoRa_controlled_variable_tests/Llama3-1B/instructive_examples_prompt/parsed_results"
conf_matrix_folder = os.path.join(parsed_folder, "confusion_matrices")
os.makedirs(conf_matrix_folder, exist_ok=True)

# Normalize labels to 'true', 'false', or 'ambiguous'
def normalize_label(val):
    if isinstance(val, str):
        val = val.strip().lower()
        if val == "true":
            return "true"
        elif val == "false":
            return "false"
    return "ambiguous"

# Process each CSV in the parsed folder
for filename in os.listdir(parsed_folder):
    if filename.endswith(".csv"):
        file_path = os.path.join(parsed_folder, filename)
        df = pd.read_csv(file_path)

        if "Ground Truth" in df.columns and "parsed_truth_value" in df.columns:
            y_true = df["Ground Truth"].apply(normalize_label)
            y_pred = df["parsed_truth_value"].apply(normalize_label)

            labels = ["true", "false", "ambiguous"]
            cm = confusion_matrix(y_true, y_pred, labels=labels)

            # Plot confusion matrix
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=labels, yticklabels=labels)
            plt.title(f"Confusion Matrix: {filename}")
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")

            # Save the plot
            plot_path = os.path.join(conf_matrix_folder, f"{os.path.splitext(filename)[0]}_confusion_matrix.png")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()

# List saved confusion matrix files
os.listdir(conf_matrix_folder)