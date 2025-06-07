import re

def extract_last_boolean(llm_output):
    """
    Extracts the last occurrence of 'True' or 'False' (case insensitive) from the given text.
    
    Args:
        llm_output (list): A list containing a single string from the LLM response.
    
    Returns:
        str: The last 'True' or 'False' found (normalized to title case).
    """
    if not llm_output or not isinstance(llm_output, list) or not llm_output[0]:
        return ""
    
    text = llm_output[0]
    
    # Search for the last occurrence of 'True' or 'False' (case insensitive)
    match = list(re.finditer(r'\b(true|false)\b', text, re.IGNORECASE))
    
    if not match:
        return ""
    
    return match[-1].group(1).title()  # Return the last match, normalized to 'True' or 'False'

# Example usage:
llm_output = ["<s>[INST]\n    Determine if the 2 words are semantically similar. Provide 'True' or 'False'. Word1:pressure, Word2:Pressure\n    [/INST]\n    > False </s>>>"]
result = extract_last_boolean(llm_output)
print(result)  # Should return 'True'
