#print(list(range(-151, 0, 10)))

from datetime import datetime, timedelta

# Assuming data_list is your data array
data_list = [
    {
        "ticker": "AAPL",
        "result": 222.67419572114943,
        "time": "2024-11-01T19:57:00",
        "resolution": "15_min",
        "real_data": 222.59
    },
    {
        "ticker": "AAPL",
        "result": 222.67550694584847,
        "time": "2024-11-01T19:58:00",
        "resolution": "15_min",
        "real_data": 222.6
    },
    {
        "ticker": "AAPL",
        "result": 222.66134012132883,
        "time": "2024-11-01T19:59:00",
        "resolution": "15_min",
        "real_data": 222.63
    }
    # Add more data here...
]
from datetime import datetime, timedelta
import json
import random

# Function to generate random testing data
def generate_test_data(num_entries=500, start_time="2024-11-01T16:32:00"):
    data_list = []
    base_time = datetime.fromisoformat(start_time)
    
    for i in range(num_entries):
        entry = {
            "ticker": "AAPL",
            "result": round(random.uniform(220, 230), 2),
            "time": (base_time + timedelta(minutes=4*i)).isoformat(),
            "resolution": "15_min",
            "real_data": round(random.uniform(220, 230), 2)
        }
        data_list.append(entry)
    
    return data_list

# Generate test data
test_data = generate_test_data()

# Convert test data to JSON for inspection
test_data_json = json.dumps(test_data, indent=2)

#print(test_data_json)

# Set the interval (15 minutes)
interval = timedelta(minutes=15)

# Define the function again to ensure it's properly used here for testing.
from datetime import datetime, timedelta

# Filter function that takes a data list and a time interval for selecting data based on time-based intervals
def filter_by_time(data, interval):
    presenting_data = []
    previous_time = None

    for entry in data:
        current_time = datetime.fromisoformat(entry["time"])
        
        # Include the entry if it's the first or if the time difference is >= 15 minutes
        if previous_time is None or current_time - previous_time >= interval:
            presenting_data.append(entry)
            previous_time = current_time

    return presenting_data

# Setting interval to 15 minutes
interval = timedelta(minutes=15)

# Run the filter function on the test data
presenting_data = filter_by_time(test_data, interval)

# Displaying a portion of the resulting data to ensure correct functionality
print(presenting_data[:10])  # Showing the first 10 entries for validation purposes

