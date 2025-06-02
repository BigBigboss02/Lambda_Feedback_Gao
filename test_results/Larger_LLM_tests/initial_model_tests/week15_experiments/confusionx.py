import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the CSV file (adjust path as needed)
df = pd.read_csv('/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/test_results/one_to_one_tests/initial_1to1_tests/week15_experiments/llm_results_1000.csv')  # Replace with the actual file path

# Ensure boolean type for label and response columns
df['label'] = df['label'].astype(str).str.lower() == 'true'
df['response'] = df['response'].astype(str).str.lower() == 'true'

# Compute confusion matrix
cm = confusion_matrix(df['label'], df['response'], labels=[True, False])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["True", "False"])

# Plot
disp.plot(cmap='Blues')
plt.title("2x2 Confusion Matrix")
# Save the figure
output_path = '/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/test_results/one_to_one_tests/initial_1to1_tests/week15_experiments/confusion_matrix.png'  # Change to your desired path
plt.savefig(output_path, dpi=300, bbox_inches='tight')
