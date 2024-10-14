# advanced_data_analyzer.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class AdvancedDataAnalyzer:
    def __init__(self, file_path):
        """
        Initialize the Data Analyzer by reading the dataset.
        Args:
            file_path (str): The file path of the CSV dataset.
        """
        self.df = pd.read_csv(file_path)
        print(f"Loaded dataset with {self.df.shape[0]} rows and {self.df.shape[1]} columns.")

    def clean_data(self):
        """
        Clean the dataset by removing missing values and duplicates.
        """
        initial_shape = self.df.shape
        self.df.dropna(inplace=True)
        self.df.drop_duplicates(inplace=True)
        final_shape = self.df.shape
        print(f"Cleaned dataset: {initial_shape[0] - final_shape[0]} rows removed.")

    def detect_outliers(self, column):
        """
        Detect outliers in a specific column using Z-scores.
        Args:
            column (str): Column name to check for outliers.
        """
        z_scores = stats.zscore(self.df[column])
        outliers = self.df[(z_scores > 3) | (z_scores < -3)]
        print(f"Found {outliers.shape[0]} outliers in {column}.")
        return outliers

    def correlation_matrix(self):
        """
        Plot the correlation matrix to visualize relationships between numerical features.
        """
        corr_matrix = self.df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        plt.show()

    def visualize_trends(self, column_x, column_y):
        """
        Plot a trend between two columns to visualize relationships over time or across data points.
        Args:
            column_x (str): The x-axis column.
            column_y (str): The y-axis column.
        """
        plt.figure(figsize=(8, 6))
        sns.lineplot(x=self.df[column_x], y=self.df[column_y], marker='o')
        plt.title(f'Trend: {column_x} vs {column_y}')
        plt.xlabel(column_x)
        plt.ylabel(column_y)
        plt.grid(True)
        plt.show()

    def descriptive_stats(self, column):
        """
        Provide detailed descriptive statistics for a specific column.
        Args:
            column (str): Column name to analyze.
        """
        stats = self.df[column].describe()
        print(f"\nDescriptive Statistics for {column}:\n{stats}\n")

    def regression_analysis(self, column_x, column_y):
        """
        Perform linear regression between two columns.
        Args:
            column_x (str): The predictor (independent variable).
            column_y (str): The response (dependent variable).
        """
        slope, intercept, r_value, p_value, std_err = stats.linregress(self.df[column_x], self.df[column_y])
        print(f"Linear Regression between {column_x} and {column_y}:")
        print(f"Slope: {slope}, Intercept: {intercept}, R-squared: {r_value**2:.3f}, p-value: {p_value:.3f}, Std Error: {std_err:.3f}")

    def save_clean_data(self, output_file):
        """
        Save the cleaned dataset to a new CSV file.
        Args:
            output_file (str): The output CSV file path.
        """
        self.df.to_csv(output_file, index=False)
        print(f"Cleaned dataset saved to {output_file}")

# Example usage
if __name__ == "__main__":
    analyzer = AdvancedDataAnalyzer('your_dataset.csv')

    # Clean the dataset
    analyzer.clean_data()

    # Detect outliers in a specific column
    outliers = analyzer.detect_outliers('Column_Name')

    # Visualize correlation between features
    analyzer.correlation_matrix()

    # Trend visualization between two columns
    analyzer.visualize_trends('Date', 'Sales')

    # Get descriptive statistics
    analyzer.descriptive_stats('Sales')

    # Perform linear regression analysis
    analyzer.regression_analysis('Advertising_Spend', 'Sales')

    # Save the cleaned dataset
    analyzer.save_clean_data('cleaned_dataset.csv')
