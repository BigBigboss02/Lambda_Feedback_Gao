import json


# Define the examples
examples = [
    {
        "original": "Give 3 examples of WSN applications. *There may be more correct answers than the ones suggested., 1. Environmental monitoring, 2. Health care monitoring, 3. Industrial process control.",
        "correct": True
    },
    {
        "original": "Give 3 examples of WSN applications. *There may be more correct answers than the ones suggested., 1. Smart agriculture, 2. Habitat monitoring.",
        "correct": False
    },
    {
        "original": "Give 3 examples of WSN applications. *There may be more correct answers than the ones suggested., 1. Disaster response systems, 2. Military surveillance, 3. Wildlife tracking.",
        "correct": True
    },
    {
        "original": "Give 3 examples of WSN applications. *There may be more correct answers than the ones suggested., 1. Smart homes, 2. Allahu Akbar 3. Environmental pollution tracking.",
        "correct": False
    },
    {
        "original": "Give 3 examples of WSN applications. *There may be more correct answers than the ones suggested., 1. Oil and gas pipeline monitoring, 2. Water quality monitoring, 3. Precision agriculture.",
        "correct": True
    },
    {
        "original": "Give 3 examples of WSN applications. *There may be more correct answers than the ones suggested., 1. Forest fire detection, 2. Urban planning, 3.Cyuka Blyat.",
        "correct": False
    },
    {
        "original": "Give 3 examples of WSN applications. *There may be more correct answers than the ones suggested., 1. Smart grid management, 2. Waste management systems, 3. Flood detection.",
        "correct": True
    },
    {
        "original": "Give 3 examples of WSN applications. *There may be more correct answers than the ones suggested., 1. Building security systems, 2. 44325ggg, 3. Crop yield monitoring.",
        "correct": False
    },
    {
        "original": "Give 3 examples of WSN applications. *There may be more correct answers than the ones suggested., 1. Weather forecasting, 2. Power line monitoring, 3. Road condition monitoring.",
        "correct": True
    },
    {
        "original": "Give 3 examples of WSN applications. *There may be more correct answers than the ones suggested., 1. Smart parking systems.",
        "correct": False
    }
]

# # Define the input
# input_data = [
#     {
#         "asr_outputs": ["The weather is nice today.", "The weather is not nice today."],
#         "speech_features": {"intonation": "neutral", "pitch": "medium", "speaking_rate": "normal"}
#     },
#     {
#         "asr_outputs": ["I am thrilled to see this.", "I am not thrilled to see this."],
#         "speech_features": {"intonation": "positive", "pitch": "high", "speaking_rate": "fast"}
#     }
# ]

# Save examples and input data to separate JSON files
with open(r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Llama-3.2-1B\promts\examples1.json", "w") as examples_file:
    json.dump(examples, examples_file, indent=4)

# with open(r"C:\Users\Malub.000\.spyder-py3\AI_project_alpha\Zhuangfei_LambdaFeedback\Llama-3.2-1B\promts\input1.json", "w") as input_file:
#     json.dump(input_data, input_file, indent=4)

print("Data saved to JSON files!")
