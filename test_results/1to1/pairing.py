# Define the lists
target_list = ['car', 'journey', 'money', 'computer', 'food', 'music', 'science', 'art', 'health']
word_list = ['automobile', 'voyage', 'cash', 'laptop', 'cuisine', 'melody', 'physics', 'painting', 'medicine']

import json

# Define the lists
target_list = ['car', 'journey', 'money', 'computer', 'food', 'music', 'science', 'art', 'health']
word_list = ['automobile', 'voyage', 'cash', 'laptop', 'cuisine', 'melody', 'physics', 'painting', 'medicine']

# Create the pairings with the desired format
pairs = [(f"({target}, {word},'false')") for target in target_list for word in word_list]

# Save the pairs to a JSON file
file_path = "/Users/zhuangfeigao/Documents/GitHub/Lambda_Feedback_Gao/test_results/1to1/pairs.json"
with open(file_path, "w") as f:
    json.dump(pairs, f, indent=2)

file_path
