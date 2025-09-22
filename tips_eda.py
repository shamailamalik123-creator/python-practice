import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Step 1: Load dataset
tips_df = pd.read_csv('tips.csv')

print("Dataset loaded successfully.")
print(f"Shape: {tips_df.shape}")

# Step 2: Initial data inspection
print("\nData Info:")
tips_df.info()

print("\nDescriptive Statistics:")
print(tips_df.describe())

print("\nMissing Values:")
print(tips_df.isnull().sum())

print("\nDuplicates:")
print(tips_df.duplicated().sum())

# Step 3: Compute tip_percentage
tips_df['tip_percentage'] = (tips_df['tip'] / tips_df['total_bill']) * 100

print("\nTip Percentage computed.")
print(f"Tip Percentage stats: {tips_df['tip_percentage'].describe()}")

# Step 4: Univariate Analysis
print("\n--- Univariate Analysis ---")

# Numerical variables histograms
numerical_cols = ['total_bill', 'tip', 'tip_percentage', 'size']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, col in enumerate(numerical_cols):
    sns.histplot(data=tips_df, x=col, kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')

plt.tight_layout()
plt.savefig('univariate_numerical.png')
plt.show()

# Categorical variables bar plots
categorical_cols = ['sex', 'smoker', 'day', 'time']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, col in enumerate(categorical_cols):
    sns.countplot(data=tips_df, x=col, ax=axes[i])
    axes[i].set_title(f'Count of {col}')

plt.tight_layout()
plt.savefig('univariate_categorical.png')
plt.show()

print("Univariate plots saved as univariate_numerical.png and univariate_categorical.png")

# Step 5: Bivariate Analysis
print("\n--- Bivariate Analysis ---")

# Scatter plot: total_bill vs tip, hue=sex
plt.figure(figsize=(8, 6))
sns.scatterplot(data=tips_df, x='total_bill', y='tip', hue='sex')
plt.title('Total Bill vs Tip by Sex')
plt.savefig('bivariate_scatter.png')
plt.show()

# Boxplots: tip_percentage by categorical variables
categorical_for_box = ['smoker', 'day', 'time', 'sex', 'size']

fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.ravel()

for i, col in enumerate(categorical_for_box):
    if i < len(axes):
        sns.boxplot(data=tips_df, x=col, y='tip_percentage', ax=axes[i])
        axes[i].set_title(f'Tip Percentage by {col}')
        axes[i].tick_params(axis='x', rotation=45)

# Hide extra subplot if odd number
if len(categorical_for_box) < len(axes):
    axes[-1].set_visible(False)

plt.tight_layout()
plt.savefig('bivariate_boxplots.png')
plt.show()

print("Bivariate plots saved as bivariate_scatter.png and bivariate_boxplots.png")

# Step 6: Multivariate Analysis
print("\n--- Multivariate Analysis ---")

# Groupby aggregates: mean tip_percentage by categories
print("\nMean Tip Percentage by Sex:")
print(tips_df.groupby('sex')['tip_percentage'].mean())

print("\nMean Tip Percentage by Smoker:")
print(tips_df.groupby('smoker')['tip_percentage'].mean())

print("\nMean Tip Percentage by Day:")
print(tips_df.groupby('day')['tip_percentage'].mean())

print("\nMean Tip Percentage by Time:")
print(tips_df.groupby('time')['tip_percentage'].mean())

print("\nMean Tip Percentage by Size:")
print(tips_df.groupby('size')['tip_percentage'].mean())

# Combinations: e.g., by sex + day
print("\nMean Tip Percentage by Sex and Day:")
print(tips_df.groupby(['sex', 'day'])['tip_percentage'].mean())

# Correlation matrix for numerical variables
numerical_cols_multi = ['total_bill', 'tip', 'tip_percentage', 'size']
corr_matrix = tips_df[numerical_cols_multi].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numerical Variables')
plt.savefig('multivariate_correlation.png')
plt.show()

print("Multivariate correlation plot saved as multivariate_correlation.png")