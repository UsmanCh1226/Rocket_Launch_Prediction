import pandas as pd
import os

CLEAN_DATA_PATH = "../data/processed/spacex_launches_clean.csv"
FEATURES_PATH = "../data/processed/spacex_features.csv"
TARGET_PATH = "../data/processed/spacex_target.csv"

def feature_engineering():
    # Load clean data
    df = pd.read_csv(CLEAN_DATA_PATH)

    # -----------------------
    # 1. Target variable
    # -----------------------
    y = df["success"]

    # -----------------------
    # 2. Feature engineering
    # -----------------------
    X = df.copy()

    # Extract year from date
    X["year"] = pd.to_datetime(X["date_utc"]).dt.year

    # Drop unused columns
    X = X.drop(columns=["date_utc", "success"])

    # -----------------------
    # 3. Handle missing values
    # -----------------------
    X["payload_mass_kg"] = X["payload_mass_kg"].fillna(
        X["payload_mass_kg"].median()
    )

    X["orbit"] = X["orbit"].fillna("Unknown")

    # -----------------------
    # 4. Encode categorical variables
    # -----------------------
    categorical_cols = ["rocket_name", "orbit", "launch_site"]
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # -----------------------
    # 5. Save outputs
    # -----------------------
    os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)

    X.to_csv(FEATURES_PATH, index=False)
    y.to_csv(TARGET_PATH, index=False)

    print("Feature engineering complete!")
    print("Feature matrix shape:", X.shape)
    print("Target shape:", y.shape)

    return X, y


if __name__ == "__main__":
    feature_engineering()
