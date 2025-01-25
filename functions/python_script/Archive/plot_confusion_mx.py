# Plot the confusion matrix and save it to a file
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("3x3 Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")

# Save the plot to a file
output_path = '/mnt/data/confusion_matrix.png'
plt.savefig(output_path)

# Show the plot
plt.show()

output_path
