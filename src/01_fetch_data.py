import requests
import json
import os

RAW_DATA_PATH = "../data/raw/spacex_launches_raw.json"

def fetch_spacex_launches():
    url = "https://api.spacexdata.com/v5/launches"
    response = requests.get(url)
    launches = response.json()
    
    # Ensure raw data folder exists
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    
    with open(RAW_DATA_PATH, "w") as f:
        json.dump(launches, f, indent=4)
    
    print(f"Saved {len(launches)} launches to {RAW_DATA_PATH}")
    return launches

if __name__ == "__main__":
    fetch_spacex_launches()
