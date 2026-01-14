import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

FEATURES_PATH = "../data/processed/spacex_features.csv"
TARGET_PATH = "../data/processed/spacex_target.csv"

def train_models():
    X = pd.read_csv(FEATURES_PATH)
    y = pd.read_csv(TARGET_PATH).squeeze()

    # -----------------------
    # 1. Clean target variable
    # -----------------------
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask].astype(int)

    assert set(y.unique()).issubset({0, 1}), f"Invalid labels: {y.unique()}"

    print("Class distribution:")
    print(y.value_counts())

    # -----------------------
    # 2. Train-test split
    # -----------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # -----------------------
    # 3. Train imbalance-aware models
    # -----------------------
    log_reg = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    log_reg.fit(X_train, y_train)

    tree = DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced",
        random_state=42
    )
    tree.fit(X_train, y_train)

    print("\nModels trained successfully with class weighting!")
    print("Training samples:", X_train.shape[0])
    print("Test samples:", X_test.shape[0])

    return log_reg, tree, X_test, y_test

if __name__ == "__main__":
    train_models()
