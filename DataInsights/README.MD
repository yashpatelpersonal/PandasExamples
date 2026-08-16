# Multiple Linear Regression on Advertising Dataset

## Overview

This project performs a multiple linear regression analysis on the classic **Advertising** dataset.

- **Target variable:** `Sales` (product sales in thousands of units)
- **Predictors:** `TV`, `Radio`, `Newspaper` advertising budgets

The workflow is implemented in an object-oriented, PEP 8–compliant Python script.

## Files

- `mlr_project.py` — main Python script with classes:
    - `DataLoader` — loads and previews the dataset
    - `DataPreprocessor` — cleans data, handles missing values and outliers
    - `MultipleLinearRegressionModel` — builds and evaluates the regression model
    - `Visualizer` — generates plots (pairplot, residual plot, coefficient barplot)
- `requirements.txt` — Python dependencies
- `outputs/` — folder created at runtime containing:
    - `pairplot.png`
    - `residuals.png`
    - `coefficients.png`

## Dataset

- Source: [Advertising dataset](https://www.statlearning.com/s/Advertising.csv)
- Loaded directly from the URL in `mlr_project.py`.

## How to Run

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
