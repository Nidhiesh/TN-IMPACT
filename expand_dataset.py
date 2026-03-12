import csv
import random
from datetime import datetime, timedelta

# Read existing dataset
existing_rows = []
with open("traffic_dataset.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        existing_rows.append(row)

print(f"Current rows: {len(existing_rows)}")

# Parse the last timestamp
last_time_str = existing_rows[-1]["timestamp"]
last_time = datetime.strptime(last_time_str, "%H:%M:%S")

# Generate new rows to reach 500 total
new_rows = []
current_time = last_time + timedelta(seconds=10)

for i in range(500 - len(existing_rows)):
    # Generate random vehicle counts (similar ranges as existing data)
    row = {
        "timestamp": current_time.strftime("%H:%M:%S"),
        "lane1_cars": str(random.randint(0, 8)),
        "lane1_bikes": str(random.randint(0, 4)),
        "lane1_buses": str(random.randint(0, 3)),
        "lane1_trucks": str(random.randint(0, 2)),
        "lane2_cars": str(random.randint(0, 8)),
        "lane2_bikes": str(random.randint(0, 4)),
        "lane2_buses": str(random.randint(0, 3)),
        "lane2_trucks": str(random.randint(0, 2)),
        "lane3_cars": str(random.randint(0, 8)),
        "lane3_bikes": str(random.randint(0, 4)),
        "lane3_buses": str(random.randint(0, 3)),
        "lane3_trucks": str(random.randint(0, 2)),
        "lane4_cars": str(random.randint(0, 8)),
        "lane4_bikes": str(random.randint(0, 4)),
        "lane4_buses": str(random.randint(0, 3)),
        "lane4_trucks": str(random.randint(0, 2)),
        "emergency_lane": "none",
        "emergency_vehicle": "none",
    }
    
    # Add occasional emergency vehicle (5% chance)
    if random.random() < 0.05:
        lanes = ["lane1", "lane2", "lane3", "lane4"]
        vehicles = ["ambulance", "fire", "police"]
        row["emergency_lane"] = random.choice(lanes)
        row["emergency_vehicle"] = random.choice(vehicles)
    
    new_rows.append(row)
    current_time += timedelta(seconds=10)

# Write combined dataset
fieldnames = ["timestamp", "lane1_cars", "lane1_bikes", "lane1_buses", "lane1_trucks",
              "lane2_cars", "lane2_bikes", "lane2_buses", "lane2_trucks",
              "lane3_cars", "lane3_bikes", "lane3_buses", "lane3_trucks",
              "lane4_cars", "lane4_bikes", "lane4_buses", "lane4_trucks",
              "emergency_lane", "emergency_vehicle"]

with open("traffic_dataset.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(existing_rows + new_rows)

print(f"Dataset expanded to {len(existing_rows) + len(new_rows)} rows")
print(f"Final timestamp: {new_rows[-1]['timestamp']}")
