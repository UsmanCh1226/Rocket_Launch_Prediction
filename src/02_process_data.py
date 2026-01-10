import pandas as pd
import requests
import json
import os

RAW_DATA_PATH = "../data/raw/spacex_launches_raw.json"
CLEAN_DATA_PATH = "../data/processed/spacex_launches_clean.csv"

def fetch_rockets():
    url = "https://api.spacexdata.com/v4/rockets"
    return {r['id']: r['name'] for r in requests.get(url).json()}

def fetch_payloads():
    url = "https://api.spacexdata.com/v4/payloads"
    return {p['id']: {'mass_kg': p.get('mass_kg'), 'orbit': p.get('orbit')} 
            for p in requests.get(url).json()}

def fetch_launchpads():
    url = "https://api.spacexdata.com/v4/launchpads"
    return {l['id']: l['name'] for l in requests.get(url).json()}

def process_spacex_data():
    # Load raw launches
    with open(RAW_DATA_PATH, "r") as f:
        launches = json.load(f)
    
    df = pd.json_normalize(launches)
    
    # Keep relevant columns
    df = df[['date_utc', 'rocket', 'payloads', 'launchpad', 'success']]
    
    # Convert date
    df['date_utc'] = pd.to_datetime(df['date_utc'])
    
    # Fetch lookup tables
    rockets = fetch_rockets()
    payloads = fetch_payloads()
    launchpads = fetch_launchpads()
    
    # Map rocket and launchpad IDs to names
    df['rocket_name'] = df['rocket'].map(rockets)
    df['launch_site'] = df['launchpad'].map(launchpads)
    
    # Extract payload info (mass, orbit)
    df['payload_mass_kg'] = df['payloads'].apply(
        lambda ids: sum(payloads[i]['mass_kg'] for i in ids if payloads[i]['mass_kg'] is not None)
    )
    df['orbit'] = df['payloads'].apply(
        lambda ids: ", ".join(set(payloads[i]['orbit'] for i in ids if payloads[i]['orbit']))
    )
    
    # Drop original IDs columns
    df = df.drop(columns=['rocket', 'payloads', 'launchpad'])
    
    # Ensure processed folder exists
    os.makedirs(os.path.dirname(CLEAN_DATA_PATH), exist_ok=True)
    
    # Save clean CSV
    df.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"Cleaned data saved to {CLEAN_DATA_PATH}")
    
    return df

if __name__ == "__main__":
    process_spacex_data()
