from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Llama-3.2-1B"  # Replace with the path to your deployed model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

input_text = "You are a math teacher, correct the response. Response: 2+2 = 5"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)  # Adjust max_length as needed
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
