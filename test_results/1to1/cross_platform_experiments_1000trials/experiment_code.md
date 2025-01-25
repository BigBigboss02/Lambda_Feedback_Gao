## Naming Rules of the Experiment Results

The naming convention for experiment results consists of four components:
1. **Model Name (lowercase)**
2. **Temperature**
3. **Max New Tokens**
4. **Instruction Type**

### 1. Model Name
The model name is straightforward and written in lowercase.  
- Example: For `llama3.1` with 8 billion parameters, the corresponding code is `llama31_8b`.

### 2. Temperature
The temperature is represented to **4 decimal places**.  
- Example: For `temperature = 0.7`, the corresponding code is `0700`.

### 3. Max New Tokens
The max new tokens value is represented to **3 significant figures**.  
- Example: For `max_new_token = 50`, the corresponding code is `050`.

### 4. Prompt Dimension
The number of outcomes expected from a prompt structure:
- True or False: `02`
- True or False of Unsure: `03`
- Higher dimensions : `04`,`05`......
  
### 5. Instruction Type
The instruction type is encoded according to the following table:
- Few-shot type: `01`
- Zero-shot instruction type: `02`
- Few-shot instruction type: `03`

### Example
For a model named `gpt4o_mini` prompted with:
- Temperature = `0.2`
- Max New Tokens = `10`
- Instruction Type = Few-shot type (`01`)

The resulting code would be:  
**`gpt4o_mini_020001001`**
