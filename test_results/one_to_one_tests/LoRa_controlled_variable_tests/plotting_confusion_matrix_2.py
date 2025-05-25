import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Set up paths
input_folder = '/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/test_results/one_to_one_tests/LoRa_controlled_variable_tests/Llama3-1B/instructive_prompt/parsed_results'
output_folder = os.path.join(input_folder, 'confusion_matrices')
os.makedirs(output_folder, exist_ok=True)

# Function to process each CSV file
def process_csv(file_path, save_path):
    try:
        df = pd.read_csv(file_path)

        # Normalize and clean string values
        df['Ground Truth'] = df['Ground Truth'].astype(str).str.strip().str.lower()
        df['parsed_truth_value'] = df['parsed_truth_value'].astype(str).str.strip().str.lower()

        # Filter out any rows with null or unexpected values
        df = df[df['Ground Truth'].isin(['true', 'false']) & df['parsed_truth_value'].isin(['true', 'false'])]

        # Skip file if no valid rows remain
        if df.empty:
            print(f"Skipped {os.path.basename(file_path)} — no valid rows")
            return

        # Convert to boolean
        df['Ground Truth'] = df['Ground Truth'].map({'true': True, 'false': False})
        df['parsed_truth_value'] = df['parsed_truth_value'].map({'true': True, 'false': False})

        # Compute and plot confusion matrix
        y_true = df['Ground Truth']
        y_pred = df['parsed_truth_value']
        labels = [True, False]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['True', 'False'])

        disp.plot(cmap='Blues')
        plt.title(f"Confusion Matrix: {os.path.basename(file_path)}")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Iterate over all CSV files
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        csv_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace('.csv', '.png'))
        process_csv(csv_path, output_path)