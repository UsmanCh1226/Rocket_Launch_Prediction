import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


DATA_PATH = "data/processed/features.csv"
TARGET_COLUMN = "success"


def load_data():
    """Load processed feature data."""
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    print("\nClass distribution:")
    print(y.value_counts())

    return X, y


def split_data(X, y):
    """Train-test split with stratification."""
    return train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )


def train_models():
    """Train Logistic Regression and Decision Tree models."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    log_reg = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=None
    )
    tree = DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced",
        random_state=42
    )

    log_reg.fit(X_train, y_train)
    tree.fit(X_train, y_train)

    print("\nModels trained successfully!")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    return log_reg, tree, X_test, y_test


def train_random_forest():
    """Train Random Forest model."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        class_weight="balanced",
        random_state=42
    )

    rf.fit(X_train, y_train)

    print("Random Forest trained successfully!")

    return rf, X_test, y_test


if __name__ == "__main__":
    train_models()
    train_random_forest()