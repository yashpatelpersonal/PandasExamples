"""
Created by : Yashkumar Patel
Created On : 08/16/2026
Created For : ANOVA Dataset

This Python script performs a full ANOVA workflow:
1. Loads dataset
2. Forms research question
3. Checks ANOVA assumptions
4. Performs one-way ANOVA
5. Conducts post-hoc Tukey HSD test
6. Generates visualizations
7. Produces summary
"""

# -----------------------------
# Import Required Libraries
# -----------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, levene
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# -----------------------------
# Class: DatasetLoader
# Loads and previews dataset
# -----------------------------
class DatasetLoader:
    def __init__(self):
        """Initializes the dataset loader."""
        self.data = None

    def load_iris(self):
        """Loads the Iris dataset from seaborn."""
        self.data = sns.load_dataset("iris")
        return self.data

    def preview(self, rows=10):
        """Returns the first N rows of the dataset."""
        return self.data.head(rows)

# -----------------------------
# Class: ANOVAAssumptions
# Checks normality & homogeneity
# -----------------------------
class ANOVAAssumptions:
    def __init__(self, data):
        """Stores dataset for assumption testing."""
        self.data = data

    def plot_histograms(self):
        """Plots histograms for petal length by species."""
        sns.histplot(data=self.data, x="petal_length", hue="species", kde=True)
        plt.title("Histogram of Petal Length by Species")
        plt.show()

    def plot_boxplots(self):
        """Plots boxplots for petal length by species."""
        sns.boxplot(data=self.data, x="species", y="petal_length")
        plt.title("Boxplot of Petal Length by Species")
        plt.show()

    def levene_test(self):
        """Performs Levene’s test for homogeneity of variances."""
        groups = [
            self.data[self.data["species"] == sp]["petal_length"]
            for sp in self.data["species"].unique()
        ]
        stat, p = levene(*groups)
        return stat, p

# -----------------------------
# Class: ANOVATest
# Performs one-way ANOVA
# -----------------------------
class ANOVATest:
    def __init__(self, data):
        """Stores dataset for ANOVA testing."""
        self.data = data

    def run_anova(self):
        """Runs one-way ANOVA using scipy.stats."""
        groups = [
            self.data[self.data["species"] == sp]["petal_length"]
            for sp in self.data["species"].unique()
        ]
        F, p = f_oneway(*groups)
        return F, p

# -----------------------------
# Class: PostHocTest
# Performs Tukey HSD
# -----------------------------
class PostHocTest:
    def __init__(self, data):
        """Stores dataset for post-hoc testing."""
        self.data = data

    def tukey_test(self):
        """Runs Tukey HSD post-hoc test."""
        tukey = pairwise_tukeyhsd(
            endog=self.data["petal_length"],
            groups=self.data["species"],
            alpha=0.05
        )
        return tukey

# -----------------------------
# Class: Visualizer
# Creates additional plots
# -----------------------------
class Visualizer:
    def __init__(self, data):
        """Stores dataset for visualization."""
        self.data = data

    def mean_plot(self):
        """Plots mean petal length by species."""
        sns.pointplot(data=self.data, x="species", y="petal_length", ci="sd")
        plt.title("Mean Petal Length by Species")
        plt.show()

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    # Load dataset
    loader = DatasetLoader()
    iris = loader.load_iris()
    print("Preview of Dataset:")
    print(loader.preview())

    # Assumption checks
    assumptions = ANOVAAssumptions(iris)
    assumptions.plot_histograms()
    assumptions.plot_boxplots()

    levene_stat, levene_p = assumptions.levene_test()
    print(f"Levene’s Test: Stat={levene_stat:.4f}, p={levene_p:.4f}")

    # Hypotheses:
    # H0: Mean petal length is equal across all species.
    # H1: At least one species has a different mean petal length.

    # Run ANOVA
    anova = ANOVATest(iris)
    F_stat, p_value = anova.run_anova()
    print(f"ANOVA Results: F={F_stat:.4f}, p={p_value:.4e}")

    # Interpretation
    if p_value < 0.05:
        print("Result: Statistically significant differences exist among species.")
        posthoc = PostHocTest(iris)
        tukey_results = posthoc.tukey_test()
        print(tukey_results)
    else:
        print("Result: No statistically significant differences found.")

    # Visualizations
    viz = Visualizer(iris)
    viz.mean_plot()

    # Summary
    print("\nSUMMARY:")
    print("""
Dataset: Iris dataset (150 samples, 3 species)
Research Question: Does mean petal length differ among species?

Methodology:
- Checked ANOVA assumptions (normality visually, Levene’s test)
- Performed one-way ANOVA
- Conducted Tukey HSD post-hoc test if significant

Results:
- ANOVA p-value < 0.05 → significant differences
- Tukey HSD shows which species differ

Reflection:
AI assisted in selecting dataset, shaping research question, guiding assumption checks,
and interpreting statistical results.
""")
