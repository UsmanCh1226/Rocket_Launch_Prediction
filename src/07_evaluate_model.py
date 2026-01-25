import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Metrics for evaluation
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    classification_report, 
    confusion_matrix
)

# Optional: if you need to split data or handle models elsewhere
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split



def evaluate_models():
    # Train base models
    log_reg, tree, X_test, y_test = train_models()

    # Train Random Forest
    rf, _, _ = train_random_forest()

    # Define models AFTER they exist
    models = {
        "Logistic Regression": log_reg,
        "Decision Tree": tree,
        "Random Forest": rf
    }

    for name, model in models.items():
        print(f"\n=== {name} ===")

        y_pred = model.predict(X_test)

        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Precision (failure):", precision_score(y_test, y_pred, pos_label=0))
        print("Recall (failure):", recall_score(y_test, y_pred, pos_label=0))

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(4, 3))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Pred Fail", "Pred Success"],
            yticklabels=["Actual Fail", "Actual Success"]
        )
        plt.title(f"{name} Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.show()
