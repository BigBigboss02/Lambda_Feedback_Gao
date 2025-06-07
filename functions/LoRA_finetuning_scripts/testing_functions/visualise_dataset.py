import datasets
from renumics import spotlight

# Load the dataset
dataset = datasets.load_dataset('Bigbigboss02/trial1', split='train')

# Display basic information
print(dataset)

# Show the first few rows
print(dataset[:5])

# Visualize the dataset with Spotlight
spotlight.show(dataset)