import time
import re

def trim_llama(data):
    """
    Extracts only the final 'Output' value corresponding to the actual input, ignoring examples or instructions.

    Args:
        data (str): The input text containing multiple entries with 'Output' fields.

    Returns:
        list: A list of final 'Output' values corresponding to actual inputs.
    """
    # Split data into sections based on iterations
    iterations = re.split(r"### Input:", data)
    outputs = []

    for section in iterations[1:]:  # Skip the first part as it doesn't contain valid iterations
        # Find the last 'Output:' in the section
        match = re.search(r"Output: (.+)", section)
        if match:
            # Add the matched output to the list
            outputs.append(match.group(1).strip())

    return outputs

def query_with_wait(api_url, headers, payload, initial_wait_time=5, retry_interval=60, max_retries=5):
    print(f"Waiting {initial_wait_time} seconds for the endpoint to load...")
    time.sleep(initial_wait_time)

    for attempt in range(max_retries):
        response = requests.post(api_url, headers=headers, json=payload)

        if response.status_code == 200:
            # Extract the assistant's message content
            response_data = response.json()
            choices = response_data.get("choices", [])
            if choices:
                assistant_message = choices[0]["message"]["content"]
                return assistant_message  # Return only the assistant's message
            else:
                print("No valid choices in response.")
                return None
        elif response.status_code == 503:  # Service Unavailable
            print(f"Attempt {attempt + 1}/{max_retries}: Model is still loading. Retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)
        else:
            print(f"Error: {response.status_code}, {response.text}")
            break
    else:
        print("Failed to get a response after multiple retries.")
        return None