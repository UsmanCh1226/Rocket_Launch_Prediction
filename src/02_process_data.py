import pandas as pd
import json
import os

RAW_DATA_PATH = "../data/raw/spacex_launches_raw.json"
CLEAN_DATA_PATH = "../data/processed/spacex_launches_clean.csv"

def process_spacex_data():
    # Load raw data
    with open(RAW_DATA_PATH, "r") as f:
        launches = json.load(f)
    
    df = pd.json_normalize(launches)
    
    # Keep relevant columns
    df = df[[
        "date_utc", "rocket", "payloads", "launchpad", "success"
    ]]
    
    # Convert date
    df["date_utc"] = pd.to_datetime(df["date_utc"])
    
    # Placeholder: expand rocket, payloads, and launchpad IDs
    df["rocket"] = df["rocket"].astype(str)
    df["launchpad"] = df["launchpad"].astype(str)
    df["payload_mass_kg"] = None
    df["orbit"] = None
    
    # TODO: Add code to map rocket IDs → names, payloads → mass & orbit, launchpads → site names
    
    # Ensure processed data folder exists
    os.makedirs(os.path.dirname(CLEAN_DATA_PATH), exist_ok=True)
    
    df.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"Cleaned data saved to {CLEAN_DATA_PATH}")
    return df

if __name__ == "__main__":
    process_spacex_data()
