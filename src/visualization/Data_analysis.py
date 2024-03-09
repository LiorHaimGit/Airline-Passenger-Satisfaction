import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def dataAnalysis(data):
    # Display the first few rows of the dataset
    print("First few rows of the dataset:")
    print(data.head())

    # Summary statistics
    print("\nSummary statistics:")
    print(data.describe())

    # Data types and missing values
    print("\nData types and missing values:")
    print(data.info())

    # Histograms of numeric features
    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
    data[numeric_features].hist(bins=20, figsize=(12, 10))
    plt.suptitle('Histograms of Numeric Features')
    plt.show()

    # Box plots of numeric features
    plt.figure(figsize=(12, 10))
    sns.boxplot(data=data[numeric_features])
    plt.title('Box plots of Numeric Features')
    plt.xticks(rotation=45)
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()

    # Pairplot (scatterplot matrix)
    sns.pairplot(data[numeric_features])
    plt.suptitle('Pairplot of Numeric Features')
    plt.show()

    # Count plots for categorical features
    categorical_features = data.select_dtypes(include=['object']).columns
    for feature in categorical_features:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature, data=data, palette='viridis')
        plt.title(f'Count plot of {feature}')
        plt.xticks(rotation=45)
        plt.show()