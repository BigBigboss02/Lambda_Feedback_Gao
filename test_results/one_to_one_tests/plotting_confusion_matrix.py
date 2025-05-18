import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load and prepare the dataset
data = pd.read_csv('/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/test_results/1to1/cross_platform_experiments_1000trials/gpt4o_mini_001000503/20250125_120351.csv')
label_mapping = {"True": 1, "False": 0, "Unsure": 2}
data['Ground Truth'] = data['Ground Truth'].map(label_mapping)
data['Response'] = data['Response'].map(label_mapping)

# Filter only the necessary columns and clean the data
filtered_data = data[['Ground Truth', 'Response']].dropna().astype(int)


# Calculate the confusion matrix
conf_matrix = confusion_matrix(filtered_data['Ground Truth'], filtered_data['Response'], labels=[2, 0, 1])  # Order: Unsure, False, True

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Unsure', 'False', 'True'], 
            yticklabels=['Unsure', 'False', 'True'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()

# Save the plot in the same base folder as the CSV file
plot_saving_path = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/test_results/1to1/cross_platform_experiments_1000trials/gpt4o_mini_001000503/20250125_120351.jpg"
plt.savefig(plot_saving_path, dpi=300)
print(f"Plot saved at {plot_saving_path}")