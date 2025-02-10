import re
from typing import Callable

# Simulating RunnableLambda
class RunnableLambda:
    def __init__(self, func: Callable):
        self.func = func

    def __call__(self, response: str):
        return self.func(response)

# Helper function to extract True/False from model response
def parse_last_boolean(response):
    matches = re.findall(r'\b(true|false)\b', response, re.IGNORECASE)
    return matches[-1].capitalize() if matches else "Unknown"

# Create a parser instance
parser = RunnableLambda(parse_last_boolean)

# Test Cases
test_responses = [
    "The result is True.",
    "This statement is false, but the last word is True.",
    "No boolean values here!",
    "FALSE and then later, true in the sentence.",
    "Completely unrelated text with no values.",
    "true false true false TRUE"
]

# Running the tests
print("Testing parse_last_boolean function:")
for i, response in enumerate(test_responses, 1):
    result = parser(response)
    print(f"Test {i}: {response}\n  -> Extracted Boolean: {result}\n")