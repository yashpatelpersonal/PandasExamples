#
# 3. The script will print evaluation metrics and generate an ROC curve image.
#
# ## Dataset
#
# The dataset includes:
# - Diagnosis (M = malignant, B = benign)
# - Mean, SE, and worst values for radius, texture, perimeter, area, smoothness, etc.
#
# ## Output
#
# - Model accuracy
# - Confusion matrix
# - ROC curve (saved as roc_curve.png)
#
# ## Author
# Yashkumar Patel
# ##Created on 08/07/2026

# main.py
# Class-based logistic regression pipeline for LAB12 breast cancer dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)


class BreastCancerModel:
    def __init__(self, excel_path):
        self.excel_path = excel_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = StandardScaler()

    def load_data(self):
        """Load and preprocess dataset."""
        self.data = pd.read_excel(self.excel_path)

        # Drop ID column if present
        if "id" in self.data.columns:
            self.data = self.data.drop(columns=["id"])

        # Encode diagnosis
        self.data["diagnosis_binary"] = self.data["diagnosis"].map({"M": 1, "B": 0})

        # Features and target
        X = self.data.drop(columns=["diagnosis", "diagnosis_binary"])
        y = self.data["diagnosis_binary"]

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

    def scale_features(self):
        """Scale numeric features."""
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def train_model(self):
        """Train logistic regression model."""
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        """Evaluate model performance."""
        y_pred = self.model.predict(self.X_test)
        y_proba = self.model.predict_proba(self.X_test)[:, 1]

        accuracy = accuracy_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_proba)

        print("\nModel Accuracy:", round(accuracy, 4))
        print("\nConfusion Matrix:\n", cm)
        print("\nROC AUC Score:", round(auc, 4))

        return y_proba

    def plot_roc(self, y_proba):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(self.y_test, y_proba):.3f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - Breast Cancer Diagnosis")
        plt.legend()
        plt.grid(True)
        plt.savefig("roc_curve.png")
        plt.show()

    def run(self):
        """Run full pipeline."""
        print("Loading data...")
        self.load_data()

        print("Scaling features...")
        self.scale_features()

        print("Training model...")
        self.train_model()

        print("Evaluating model...")
        y_proba = self.evaluate()

        print("Plotting ROC curve...")
        self.plot_roc(y_proba)


if __name__ == "__main__":
    model = BreastCancerModel("Cancer.xls")
    model.run()


