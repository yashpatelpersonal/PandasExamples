# -----------------------------------------------------------
#Created By : Yashkumar Patel
#Created On: 08/05/2026
# Lab 11 – Linear Regression using Classes and Functions
# Dataset: Cancer_reg.xlsx

# -----------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# -----------------------------------------------------------
# CLASS: CancerRegressionModel
# Purpose: Load data, train model, evaluate model
# -----------------------------------------------------------
# -----------------------------------------------------------
# Visualization Add‑On for Cancer Regression Model (Excel Version)
# -----------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

class CancerRegressionModel:

    def __init__(self, excel_path):
        """Load dataset from Excel workbook (.xls or .xlsx)."""
        # openpyxl handles .xlsx, xlrd handles .xls
        self.data = pd.read_excel(excel_path)
        self.model = LinearRegression()

    def prepare_data(self, feature_cols, target_col):
        """Prepare features and target, split into train/test."""
        X = self.data[feature_cols]
        y = self.data[target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def train(self):
        """Train the regression model."""
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        """Return R² score."""
        predictions = self.model.predict(self.X_test)
        return r2_score(self.y_test, predictions)

    def visualize(self, feature_cols, target_col):
        """Generate model visualizations."""
        predictions = self.model.predict(self.X_test)

        # -----------------------------------------------------------
        # 1. Actual vs Predicted Scatter Plot
        # -----------------------------------------------------------
        plt.figure(figsize=(8, 6))
        plt.scatter(self.y_test, predictions, alpha=0.6)
        plt.xlabel("Actual Death Rate")
        plt.ylabel("Predicted Death Rate")
        plt.title("Actual vs Predicted Values")
        plt.grid(True)
        plt.show()

        # -----------------------------------------------------------
        # 2. Residual Plot
        # -----------------------------------------------------------
        residuals = self.y_test - predictions
        plt.figure(figsize=(8, 6))
        plt.scatter(predictions, residuals, alpha=0.6)
        plt.axhline(0, color="red", linestyle="--")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        plt.grid(True)
        plt.show()

        # -----------------------------------------------------------
        # 3. Correlation Heatmap
        # -----------------------------------------------------------
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.data[feature_cols + [target_col]].corr(),
                    annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.show()

        # -----------------------------------------------------------
        # 4. Feature Coefficient Bar Chart
        # -----------------------------------------------------------
        coef_values = self.model.coef_
        plt.figure(figsize=(8, 6))
        sns.barplot(x=feature_cols, y=coef_values)
        plt.title("Feature Impact (Regression Coefficients)")
        plt.ylabel("Coefficient Value")
        plt.xticks(rotation=45)
        plt.show()


def run_regression():
    """Run full workflow including visualizations."""
    model = CancerRegressionModel("cancer_reg.xls")  # <-- Excel file here

    feature_cols = ["incidenceRate", "medIncome", "povertyPercent"]
    target_col = "TARGET_deathRate"

    model.prepare_data(feature_cols, target_col)
    model.train()

    print("R² Score:", model.evaluate())
    model.visualize(feature_cols, target_col)


if __name__ == "__main__":
    run_regression()
