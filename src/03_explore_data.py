import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CLEAN_DATA_PATH = "../data/processed/spacex_launches_clean.csv"

def explore_data():
    # Load the clean dataset
    df = pd.read_csv(CLEAN_DATA_PATH)
    
    print("=== Dataset Overview ===")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Missing values per column:\n", df.isnull().sum())
    
    print("\n=== Sample Data ===")
    print(df.head())

    # --- Basic Statistics ---
    print("\n=== Payload Mass Stats ===")
    print(df['payload_mass_kg'].describe())
    
    # --- Success Rate ---
    print("\n=== Launch Success Rate ===")
    print(df['success'].value_counts(normalize=True))

    # --- Plots ---
    sns.set(style="whitegrid")
    
    # 1. Success vs Rocket
    plt.figure(figsize=(8,5))
    sns.countplot(data=df, x='rocket_name', hue='success')
    plt.title("Launch Success by Rocket")
    plt.xlabel("Rocket")
    plt.ylabel("Number of Launches")
    plt.xticks(rotation=45)
    plt.legend(title='Success')
    plt.tight_layout()
    plt.show()
    
    # 2. Payload Mass Distribution
    plt.figure(figsize=(8,5))
    sns.histplot(df['payload_mass_kg'].dropna(), bins=20, kde=True)
    plt.title("Payload Mass Distribution")
    plt.xlabel("Payload Mass (kg)")
    plt.ylabel("Count")
    plt.show()
    
    # 3. Success Rate Over Time
    df['year'] = pd.to_datetime(df['date_utc']).dt.year
    yearly_success = df.groupby('year')['success'].mean()
    
    plt.figure(figsize=(10,5))
    sns.lineplot(x=yearly_success.index, y=yearly_success.values, marker='o')
    plt.title("Yearly Launch Success Rate")
    plt.xlabel("Year")
    plt.ylabel("Success Rate")
    plt.ylim(0, 1.05)
    plt.show()
    
    print("\n=== Done with EDA ===")

if __name__ == "__main__":
    explore_data()
