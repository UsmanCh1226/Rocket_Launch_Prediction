import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

FEATURES_PATH = "../data/processed/spacex_features.csv"
TARGET_PATH = "../data/processed/spacex_target.csv"

def train_random_forest():
    # Load data
    X = pd.read_csv(FEATURES_PATH)
    y = pd.read_csv(TARGET_PATH).squeeze()

    # Clean target
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask].astype(int)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,      # number of trees
        max_depth=6,           # prevents overfitting
        class_weight="balanced",
        random_state=42
    )

    rf.fit(X_train, y_train)

    print("Random Forest trained successfully!")

    return rf, X_test, y_test


if __name__ == "__main__":
    train_random_forest()
