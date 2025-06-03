# Lambda_Feedback_Gao

This is the repository of Gao Zhuangfei's Final Year Project.

## 📦 Setup
1. Install all dependencies from `requirements.txt`.

## 📁 Repository Overview

- `functions/`: All executable modules including Python scripts, PostgreSQL scripts, and LoRA training components.
- `test_results/`: Outputs from evaluation runs on various large language models (LLMs), including both baseline and LoRA-augmented tests.
- `non_functions/`: Supplementary resources such as notes, version control records, and environment setup scripts.
- `shortTextAnswer/`: Forked branch from the Lambda Feedback repo — refer to its internal `README` for app-specific instructions.
- `docs/`: Documentation files related to the project.

## 🔗 External Resources

- HuggingFace datasets and models:  
  👉 [https://huggingface.co/Bigbigboss02](https://huggingface.co/Bigbigboss02)

---

## 📂 Project Folder Structure
<pre lang="markdown">
```
Lambda_Feedback_Gao/
├── docs/
├── shortTextAnswer/                  # Forked Lambda Feedback branch, see internal README
│   ├── app/
│   └── .github/
├── non_functions/                    # Non-experiment resources
│   ├── version_control/
│   ├── cuda_env_test/
│   └── notes_and_literatures/
├── functions/                        # Experiment-related code and modules
│   ├── postgreSQL_script/           # Low value — initial SQL scripts
│   │   └── tests/
│   ├── python_script/               # Moderate value — core Python scripts
│   │   ├── tools/
│   │   ├── Archive/                 # Low value — archived versions
│   │   ├── tests/                   # Moderate value — test functions
│   │   ├── Evaluation_functions/    # Low value — early LLM evals
│   │   ├── structured_prompts/      # Low value — prompt logic separation
│   │   │   ├── LongChain/
│   │   │   └── confusion_matrix/
│   │   └── Severless_API_calls/     # Moderate value — GPT-4o-mini endpoint calls
│   ├── LoRa/                        # High value — training/testing LoRA modules
│   │   ├── calling_functions/       # High value — LoRA model interface
│   │   ├── adaptors/                # High/Moderate value — trained LoRA weights
│   │   │   ├── tuned_Llama321B_adaptors/
│   │   │   ├── tuned_BeRT_adaptors/
│   │   │   ├── tuned_Llama321B_balanced_dataset/
│   │   │   └── selected_Llama321B_adaptors/
│   │   ├── testing_functions/       # Moderate value
│   │   └── data/                    # High value — training/testing sets
│   │       └── waste_data/          # Low value
│   └── google_colab_script/         # High value — LoRA training notebook
├── test_results/                    # High value — model test results, includes everything in Paper Results
│   ├── Smaller_LLM_tests/
│   │   ├── LoRa_controlled_variable_tests/
│   │   ├── Base_controlled_variable_tests/
│   │   ├── Llama3_1B_initial_test/
│   │   └── initial_finetuned_model_test/
│   ├── Larger_LLM_tests/
│   │   ├── Base_gpt4o_llama3_crossplatform/
│   │   ├── initial_base_model_tests/
│   │   │   ├── week15_experiments/
│   │   │   └── week16_experiments/
│   │   └── Cross_Platform_Comparison/
│   │       ├── Llama3_3B_test/
│   │       ├── GPT-4o_mini_8B_test/
│   │       └── Llama3_1B_test/
```
</pre>
