import pandas as pd
CLEAN_DATA_PATH = "../data/processed/spacex_launches_clean.csv"

def explore_data():
    df = pd.read_csv(CLEAN_DATA_PATH)
    
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Sample:")
    print(df.head())
    
    # Quick stats
    print("\nSuccess rate:")
    print(df["success"].value_counts(normalize=True))
    
    # Optional: simple plot
    try:
        import matplotlib.pyplot as plt
        df["success"].value_counts().plot(kind="bar", title="Launch Success vs Failure")
        plt.show()
    except:
        pass

if __name__ == "__main__":
    explore_data()
