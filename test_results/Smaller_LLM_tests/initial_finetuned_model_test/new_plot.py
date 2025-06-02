import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def extract_prediction(text):
    try:
        # Get the last [/INST]
        match = re.split(r'\[/INST\]', text)[-1]
        # Search for 'True' or 'False' (case insensitive)
        pred_match = re.search(r'\b(True|False)\b', match, re.IGNORECASE)
        if pred_match:
            return pred_match.group(1).capitalize()
    except Exception as e:
        print(f"Error processing text: {text}\n{e}")
    return None

def plot_confusion_matrix_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    y_true = []
    y_pred = []

    for _, row in df.iterrows():
        ground_truth = str(row['Ground Truth']).capitalize()
        model_output = row['Model Output']
        prediction = extract_prediction(model_output)

        if prediction in ['True', 'False']:
            y_true.append(ground_truth)
            y_pred.append(prediction)
        else:
            # Skipping row with no valid prediction
            continue

    cm = confusion_matrix(y_true, y_pred, labels=['True', 'False'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['True', 'False'])
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

# Example usage
csv_path = '/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/test_results/1to1/finetuned_model_test/output_results_lora_full.csv'  # Replace with your actual CSV path
plot_confusion_matrix_from_csv(csv_path)