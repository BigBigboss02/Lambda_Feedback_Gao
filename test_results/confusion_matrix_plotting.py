import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Sample CSV data for testing
data = pd.DataFrame({
    'target': np.random.choice(['Class1', 'Class2', 'Class3'], size=100, p=[0.3, 0.4, 0.3]),
    'result': np.random.choice(['Class1', 'Class2', 'Class3'], size=100, p=[0.3, 0.4, 0.3])
})

# Save the sample data to CSV (if needed for external testing)
# data.to_csv('sample_data.csv', index=False)

# Calculate the confusion matrix
target = data['target']
result = data['result']
cm = confusion_matrix(target, result, labels=['Class1', 'Class2', 'Class3'])

# Visualize the confusion matrix
cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class1', 'Class2', 'Class3'])

# Customize and plot
cmd.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Additional information on incorrect predictions
incorrect_count = (target != result).sum()
total_count = len(data)
percentage_incorrect = (incorrect_count / total_count) * 100

print(f"Total Rows: {total_count}")
print(f"Incorrect Predictions: {incorrect_count}")
print(f"Percentage Incorrect: {percentage_incorrect:.2f}%")