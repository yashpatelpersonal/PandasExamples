# Cancer Regression – Linear Regression Model

## Overview
This project performs linear regression on the Cancer_reg.csv dataset to predict TARGET_deathRate using selected demographic features.  
The code uses classes, functions, and detailed comments for clarity.

## Project Structure
project-folder/
│
├── cancer_regression.py     # Main Python script
├── Cancer_reg.xlsx           # Dataset
├── requirements.txt         # Python dependencies
└── README.md                # Documentation

## Installation
Install required packages:

pip install -r requirements.txt

## Excel / Workbook Support
This project includes support for reading Excel workbooks using pandas.

The dependency **openpyxl** is included in requirements.txt so you can load Excel files like:

import pandas as pd
df = pd.read_excel("Cancer_reg.xlsx")

## Running the Program
Run the script:

python cancer_regression.py

## Features Used in Regression
Predictors:
- incidenceRate
- medIncome
- povertyPercent

Target:
- TARGET_deathRate

## Code Structure
### Class: CancerRegressionModel
Handles:
- Loading data (CSV or Excel)
- Preparing features
- Training the model
- Evaluating performance
- Displaying coefficients

### Function: run_regression()
Controls the full workflow.

## Example Output
Model Evaluation:
R² Score: 0.62

Regression Coefficients:
incidenceRate: 0.1452
medIncome: -0.0018
povertyPercent: 0.7541

## Interpretation
Higher incidenceRate → higher predicted death rate  
Higher medIncome → slightly lower death rate  
Higher povertyPercent → significantly higher death rate

## Requirements
See requirements.txt for all dependencies.
