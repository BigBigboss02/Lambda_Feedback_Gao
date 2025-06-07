# Lambda_Feedback_Gao

This is the repository of Gao Zhuangfei's Final Year Project.

## 📦 Setup
1. Install all dependencies from `requirements_vs3.txt`.
2. Set up your login_configs.env following the format of pseudo_login_configs.env
3. Download necessary base models from Huggingface (see External Resources)
4. Set the folder paths in the funtion you are using to your local folder location 
5. Local NVDIA GPU or Apple silican might be needed, situation depending on the function you are using


## 📁 Repository Overview

- `functions/`: All executable modules including Python scripts, PostgreSQL scripts, and LoRA training components.
- `test_results/`: Outputs from evaluation runs on various large language models (LLMs), including both baseline and LoRA-augmented tests.
- `non_functions/`: Supplementary resources such as notes, version control records, and environment setup scripts.
- `shortTextAnswer/`: Forked branch from the Lambda Feedback repo — refer to its internal `README` for app-specific instructions.

## 🔗 External Resources
- Huggingface base models to download and use:  
  👉 [LLaMA-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)  
  👉 [LLaMA-3.1-8B](https://huggingface.co/Bigbigboss02/llama3.1-8B)  
  👉 [BERT Large Uncased](https://huggingface.co/Bigbigboss02/bert-large-uncased)  
- HuggingFace datasets and models for Colab integration:  
  👉 [https://huggingface.co/Bigbigboss02](https://huggingface.co/Bigbigboss02)

---

## 📂 Project Folder Structure
<pre lang="markdown">
```
Lambda_Feedback_Gao/

├── non_functions/                    # Non-experiment resources
│   ├── version_control/
│   ├── cuda_env_test/
│   └── notes_and_literatures/

├── functions/                        # Experiment-related code and modules
│   ├── google_colab_scripts/        # High value — LoRA training notebook
│   ├── postgreSQL_scripts/          # Low value — initial SQL scripts
│   │   └── tests/
│   ├── LoRA_finetuning_scripts/     # High value — LoRA training/testing
│   │   ├── calling_functions/       # High value — LoRA model interface
│   │   ├── adaptors/                # High/Moderate value — trained LoRA weights
│   │   │   ├── tuned_Llama321B_adaptors/
│   │   │   ├── tuned_BeRT_adaptors/
│   │   │   ├── tuned_Llama321B_balanced_dataset/
│   │   │   └── selected_Llama321B_adaptors/
│   │   ├── testing_functions/       # Moderate value
│   │   └── data/
│   │       └── waste_data/          # Low value
│   └── Prompt_Engineering_scripts/  # Moderate value — prompt logic + API
│       ├── Archive/                 # Low value — archived code
│       ├── __pycache__/
│       ├── Initial_experiments/
│       │   ├── tools/
│       │   │   └── __pycache__/
│       │   ├── tests/
│       │   └── structured_prompts/
│       │       ├── LongChain/
│       │       └── confusion_matrix/
│       └── Severless_API_calls/     # Moderate value — GPT-4o-mini endpoint calls

├── test_results/                    # High value — model test results
│   ├── Larger_LLM_tests/
│   │   ├── one_to_many_Cross_Platform_Comparison/
│   │   │   ├── Llama3_3B_test/
│   │   │   ├── GPT-4o_mini_8B_test/
│   │   │   └── Llama3_1B_test/
│   │   ├── one_to_one_Cross_Platform_Comparison/
│   │   │   └── [various GPT-4o-mini and Llama3.2-1B result folders]
│   │   └── initial_model_tests/
│   │       ├── week15_experiments/
│   │       └── week16_experiments/
│   ├── Wasted_Data/
│   └── Smaller_LLM_tests/
│       ├── LoRa_controlled_variable_tests/
│       │   ├── BeRT_large/                             # BeRT testing results
│       │   │   └── confusion_matrices/
│       │   └── Llama3-1B_balanced/                     # Trained on Balanced training dataset
│       │       ├── instructive_prompt/                 # Named by prompt type used at testing
│       │       │   │                                   # Csv file named prompt/arg type used at fine-tuning
│       │       │   └── parsed_results/                 # Parsed results
│       │       │       └── confusion_matrices/         # Confusion matrices ploted
│       │       └── instructive_examples_prompt/
│       │           └── parsed_results/
│       │               └── confusion_matrices/
│       │   └── Llama3-1B_main/                         # Trained on Main training dataset
│       │       ├── balanced_training_material_prompt/
│       │       │   ├── instructive_prompt/
│       │       │   │   └── parsed_results/
│       │       │   │       └── confusion_matrices/
│       │       │   └── instructive_examples_prompt/
│       │       │       └── parsed_results/
│       │       │           └── confusion_matrices/
│       │       ├── main_training_material_prompt/
│       │       │   ├── instructive_prompt/
│       │       │   │   └── parsed_results/
│       │       │   │       └── confusion_matrices/
│       │       │   ├── examples_prompt/
│       │       │   │   └── parsed_results/
│       │       │   │       └── confusion_matrices/
│       │       │   ├── instructive_examples_prompt/
│       │       │   │   └── parsed_results/
│       │       │   │       └── confusion_matrices/
│       │       │   └── null_prompt/
│       │       │       └── parsed_results/
│       │       │           └── confusion_matrices/
│       │       └── new_testing_material_prompt/
│       │           ├── instructive_prompt/
│       │           │   └── parsed_results/
│       │           │       └── confusion_matrices/
│       │           └── instructive_examples_prompt/
│       │               └── parsed_results/
│       │                   └── confusion_matrices/
│       ├── Base_controlled_variable_tests/
│       │   ├── BeRT_large/
│       │   │   └── confusion_matrices/
│       │   └── llama32_1b/
│       │       └── parsed_results/
│       │           └── confusion_matrices/
│       ├── Llama3_1B_initial_test/
│       └── initial_finetuned_model_test/ # Low value, initial attempts

```
</pre>

