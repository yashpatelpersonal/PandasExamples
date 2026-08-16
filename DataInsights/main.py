"""
Multiple Linear Regression Project using Advertising dataset.

Dataset source:
https://www.statlearning.com/s/Advertising.csv

Research question:
How do TV, Radio, and Newspaper advertising budgets jointly predict product sales?

This script:
- Loads data into a Pandas DataFrame
- Cleans and prepares data
- Builds a multiple linear regression model (statsmodels)
- Evaluates model (R², adjusted R², p-values)
- Visualizes relationships and residuals

PEP 8 compliant, object-oriented structure.
"""

import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm


class DataLoader:
    """Class responsible for loading and previewing the dataset."""

    def __init__(self, url: str) -> None:
        """
        Initialize DataLoader with dataset URL.

        Parameters
        ----------
        url : str
            URL to the CSV dataset.
        """
        self.url = url
        self.df: pd.DataFrame | None = None

    def load_data(self) -> pd.DataFrame:
        """
        Load dataset from URL into a Pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            Loaded dataset.
        """
        self.df = pd.read_csv(self.url)
        return self.df

    def preview(self, n_rows: int = 10) -> pd.DataFrame:
        """
        Return the first n_rows of the dataset.

        Parameters
        ----------
        n_rows : int
            Number of rows to preview.

        Returns
        -------
        pd.DataFrame
            Preview of the dataset.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.df.head(n_rows)


class DataPreprocessor:
    """Class responsible for cleaning and preparing the dataset."""

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize DataPreprocessor with a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Raw dataset.
        """
        self.df = df.copy()

    def convert_types(self) -> pd.DataFrame:
        """
        Ensure numeric columns are of numeric dtype.

        Returns
        -------
        pd.DataFrame
            DataFrame with converted types.
        """
        numeric_cols = ["TV", "radio", "newspaper", "sales"]
        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        return self.df

    def handle_missing_values(self) -> pd.DataFrame:
        """
        Handle missing values by dropping rows with any NaNs.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame.
        """
        self.df = self.df.dropna()
        return self.df

    def handle_outliers(self) -> pd.DataFrame:
        """
        Basic outlier handling using IQR rule for numeric columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with outliers removed.
        """
        numeric_cols = ["TV", "radio", "newspaper", "sales"]
        for col in numeric_cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            self.df = self.df[
                (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)
                ]
        return self.df

    def get_clean_data(self) -> pd.DataFrame:
        """
        Run full preprocessing pipeline.

        Returns
        -------
        pd.DataFrame
            Fully cleaned DataFrame.
        """
        self.convert_types()
        self.handle_missing_values()
        self.handle_outliers()
        return self.df


class MultipleLinearRegressionModel:
    """Class for building and evaluating a multiple linear regression model."""

    def __init__(self, df: pd.DataFrame, predictors: List[str], target: str) -> None:
        """
        Initialize the regression model.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned dataset.
        predictors : List[str]
            List of predictor column names.
        target : str
            Target column name.
        """
        self.df = df
        self.predictors = predictors
        self.target = target
        self.model = None
        self.results = None

    def build_model(self) -> sm.regression.linear_model.RegressionResultsWrapper:
        """
        Build and fit the multiple linear regression model using statsmodels.

        Returns
        -------
        RegressionResultsWrapper
            Fitted model results.
        """
        x = self.df[self.predictors]
        x = sm.add_constant(x)
        y = self.df[self.target]
        self.model = sm.OLS(y, x)
        self.results = self.model.fit()
        return self.results

    def print_summary(self) -> None:
        """
        Print the model summary including coefficients, R², and p-values.
        """
        if self.results is None:
            raise ValueError("Model not built. Call build_model() first.")
        print(self.results.summary())

    @property
    def r_squared(self) -> float:
        """
        Return R² value.

        Returns
        -------
        float
            R² value.
        """
        if self.results is None:
            raise ValueError("Model not built. Call build_model() first.")
        return float(self.results.rsquared)

    @property
    def adjusted_r_squared(self) -> float:
        """
        Return adjusted R² value.

        Returns
        -------
        float
            Adjusted R² value.
        """
        if self.results is None:
            raise ValueError("Model not built. Call build_model() first.")
        return float(self.results.rsquared_adj)

    def get_coefficients(self) -> pd.DataFrame:
        """
        Return model coefficients and p-values.

        Returns
        -------
        pd.DataFrame
            DataFrame with coefficients and p-values.
        """
        if self.results is None:
            raise ValueError("Model not built. Call build_model() first.")
        coef_df = pd.DataFrame(
            {
                "coef": self.results.params,
                "p_value": self.results.pvalues,
            }
        )
        return coef_df


class Visualizer:
    """Class for creating visualizations for the regression analysis."""

    def __init__(self, df: pd.DataFrame, results: sm.regression.linear_model.RegressionResultsWrapper) -> None:
        """
        Initialize Visualizer.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned dataset.
        results : RegressionResultsWrapper
            Fitted regression model results.
        """
        self.df = df
        self.results = results

    def pairplot(self, output_path: str = "pairplot.png") -> None:
        """
        Create a pairplot of predictors and target.

        Parameters
        ----------
        output_path : str
            File path to save the plot.
        """
        sns.pairplot(self.df[["TV", "radio", "newspaper", "sales"]])
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def residual_plot(self, output_path: str = "residuals.png") -> None:
        """
        Plot residuals vs fitted values.

        Parameters
        ----------
        output_path : str
            File path to save the plot.
        """
        fitted_vals = self.results.fittedvalues
        residuals = self.results.resid

        plt.figure(figsize=(8, 6))
        plt.scatter(fitted_vals, residuals, alpha=0.7)
        plt.axhline(0, color="red", linestyle="--")
        plt.xlabel("Fitted values")
        plt.ylabel("Residuals")
        plt.title("Residuals vs Fitted Values")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def coefficient_barplot(self, output_path: str = "coefficients.png") -> None:
        """
        Plot coefficients as a bar chart.

        Parameters
        ----------
        output_path : str
            File path to save the plot.
        """
        coef_df = pd.DataFrame(
            {
                "variable": self.results.params.index,
                "coef": self.results.params.values,
            }
        )
        coef_df = coef_df[coef_df["variable"] != "const"]

        plt.figure(figsize=(8, 6))
        sns.barplot(x="variable", y="coef", data=coef_df)
        plt.title("Regression Coefficients")
        plt.xlabel("Predictor")
        plt.ylabel("Coefficient")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


def main() -> None:
    """
    Main workflow:
    - Load data
    - Preview first 10 rows (printed to console)
    - Preprocess data
    - Build and evaluate model
    - Generate visualizations
    """
    url = "https://www.statlearning.com/s/Advertising.csv"

    # Load data
    loader = DataLoader(url=url)
    df_raw = loader.load_data()
    print("First 10 rows of the dataset:")
    print(loader.preview(10))

    # Preprocess data
    preprocessor = DataPreprocessor(df=df_raw)
    df_clean = preprocessor.get_clean_data()

    # Define predictors and target
    predictors = ["TV", "radio", "newspaper"]
    target = "sales"

    # Build model
    mlr_model = MultipleLinearRegressionModel(
        df=df_clean,
        predictors=predictors,
        target=target,
    )
    results = mlr_model.build_model()
    mlr_model.print_summary()

    print(f"R²: {mlr_model.r_squared:.4f}")
    print(f"Adjusted R²: {mlr_model.adjusted_r_squared:.4f}")
    print("Coefficients and p-values:")
    print(mlr_model.get_coefficients())

    # Create output directory
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Visualizations
    visualizer = Visualizer(df=df_clean, results=results)
    visualizer.pairplot(output_path=os.path.join(output_dir, "pairplot.png"))
    visualizer.residual_plot(output_path=os.path.join(output_dir, "residuals.png"))
    visualizer.coefficient_barplot(
        output_path=os.path.join(output_dir, "coefficients.png")
    )


if __name__ == "__main__":
    main()
