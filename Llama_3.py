import csv
import re
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
input_text = """
Task: Correct the following responses. Provide 10 examples of WSN applications as output. Do not include the input text or any additional context in the output.

Examples:

1. Original: "Give 3 examples of WSN applications. *There may be more correct answers than the ones suggested.,"1. Environmental monitoring, 2. Health care monitoring, 3. Industrial process control."
   Corrected: "1. Environmental monitoring, 2. Health care monitoring, 3. Industrial process control."

2. Original: "Give 3 examples of WSN applications. *There may be more correct answers than the ones suggested.,"1. Smart agriculture, 2. Structural health monitoring, 3. Habitat monitoring."
   Corrected: "1. Smart agriculture, 2. Structural health monitoring, 3. Habitat monitoring."

3. Original: "Give 3 examples of WSN applications. *There may be more correct answers than the ones suggested.,"1. Disaster response systems, 2. Military surveillance, 3. Wildlife tracking."
   Corrected: "1. Disaster response systems, 2. Military surveillance, 3. Wildlife tracking."

4. Original: "Give 3 examples of WSN applications. *There may be more correct answers than the ones suggested.,"1. Smart homes, 2. Traffic monitoring, 3. Environmental pollution tracking."
   Corrected: "1. Smart homes, 2. Traffic monitoring, 3. Environmental pollution tracking."

5. Original: "Give 3 examples of WSN applications. *There may be more correct answers than the ones suggested.,"1. Oil and gas pipeline monitoring, 2. Water quality monitoring, 3. Precision agriculture."
   Corrected: "1. Oil and gas pipeline monitoring, 2. Water quality monitoring, 3. Precision agriculture."

6. Original: "Give 3 examples of WSN applications. *There may be more correct answers than the ones suggested.,"1. Forest fire detection, 2. Urban planning, 3. Air quality monitoring."
   Corrected: "1. Forest fire detection, 2. Urban planning, 3. Air quality monitoring."

7. Original: "Give 3 examples of WSN applications. *There may be more correct answers than the ones suggested.,"1. Smart grid management, 2. Waste management systems, 3. Flood detection."
   Corrected: "1. Smart grid management, 2. Waste management systems, 3. Flood detection."

8. Original: "Give 3 examples of WSN applications. *There may be more correct answers than the ones suggested.,"1. Building security systems, 2. Patient health monitoring, 3. Crop yield monitoring."
   Corrected: "1. Building security systems, 2. Patient health monitoring, 3. Crop yield monitoring."

9. Original: "Give 3 examples of WSN applications. *There may be more correct answers than the ones suggested.,"1. Weather forecasting, 2. Power line monitoring, 3. Road condition monitoring."
   Corrected: "1. Weather forecasting, 2. Power line monitoring, 3. Road condition monitoring."

10. Original: "Give 3 examples of WSN applications. *There may be more correct answers than the ones suggested.,"1. Asset tracking, 2. Energy usage monitoring, 3. Smart parking systems."
    Corrected: "1. Asset tracking, 2. Energy usage monitoring, 3. Smart parking systems."

Instructions: Use the same format as in the examples. Provide 10 examples of WSN applications as output for the input below.

Input:
"Give 3 examples of WSN applications. *There may be more correct answers than the ones suggested."


Output:
"""
"""
- Question: 3+3=7. Correction: 3+3=6.
- Question: 10-5=2. Correction: 10-5=5.
- Question: 8-3=6. Correction: 8-3=5.
- Question: 2+2=5. Correction:
"""
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# Filepath for the CSV file, named by the current system time
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filepath = rf"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Lambda_Feedback_Gao\result\Llama3_test\llama3_{current_time}.csv"

# Open the CSV file to record results
with open(filepath, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Iteration", "Response"])

    # Run the prompt 100 times and record the result
    for i in range(1, 10):
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,  # Limit response length
            temperature=0.7,  # Balance creativity and determinism
            top_p=0.9,  # Encourage high-probability tokens
            num_return_sequences=1  # Generate a single output
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # # Extract the corrected outputs using regex
        # corrected_outputs = re.findall(r'Corrected:\s*"(.*?)"', response)

        # # Format the corrected outputs into a clean list
        # formatted_output = "\n".join(corrected_outputs)


        print(f"Iteration {i}: {response}")
        writer.writerow([i, response])

print(f"Results have been recorded to {filepath}")
