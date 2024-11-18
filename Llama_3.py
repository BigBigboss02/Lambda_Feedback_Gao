import csv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datetime import datetime

# Specify your model path
model_name = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Llama-3.2-1B"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Move model to the appropriate device (GPU or CPU)
model = model.to(device)

# Prepare the input
input_text = """Correct the responses. Provide only corrected statements:
- Question: 3+3=7. Correction: 3+3=6.
- Question: 10-5=2. Correction: 10-5=5.
- Question: 8-3=6. Correction: 8-3=5.
- Question: 2+2=5. Correction:
"""

inputs = tokenizer(input_text, return_tensors="pt").to(device)

# Filepath for the CSV file, named by the current system time
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filepath = rf"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Lambda_Feedback_Gao\result\llama3_{current_time}.csv"

# Open the CSV file to record results
with open(filepath, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Iteration", "Response"])

    # Run the prompt 100 times and record the result
    for i in range(1, 10):
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,  # Limit response length
            temperature=0.7,  # Balance creativity and determinism
            top_p=0.9,  # Encourage high-probability tokens
            num_return_sequences=1  # Generate a single output
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Iteration {i}: {response}")
        writer.writerow([i, response])

print(f"Results have been recorded to {filepath}")
