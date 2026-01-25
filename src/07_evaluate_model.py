import pandas as pd
from src.random_forest_model import train_random_forest


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Correct import
from src.train_model import train_models  # use the renamed file

def evaluate_models():
    log_reg, tree, X_test, y_test = train_models()

    rf, X_test_rf, y_test_rf = train_random_forest()
    models["Random Forest"] = rf

    models = {
        "Logistic Regression": log_reg,
        "Decision Tree": tree
    }

    for name, model in models.items():
        print(f"\n=== {name} ===")

        y_pred = model.predict(X_test)

        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Precision (failure):", precision_score(y_test, y_pred, pos_label=0))
        print("Recall (failure):", recall_score(y_test, y_pred, pos_label=0))

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Confusion matrix
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


if __name__ == "__main__":
    evaluate_models()
