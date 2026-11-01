# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the Datasets
# Ensure 'Unemployment in India.csv' is in the working directory.
df = pd.read_csv('Unemployment in India.csv')

# Clean column names by stripping trailing/leading whitespaces
df.columns = df.columns.str.strip()

print("Columns in dataset:", df.columns)
print(df.head())

# 2. Data Cleaning & Preprocessing
# Inspect missing values
print("\nMissing values per column:\n", df.isnull().sum())

# Drop rows with missing values if any exist
df = df.dropna()

# Rename the columns present in this dataset for convenience.
df = df.rename(columns={
    'Region': 'State',
    'Estimated Unemployment Rate (%)': 'Unemployment_Rate',
    'Estimated Employed': 'Employed',
    'Estimated Labour Participation Rate (%)': 'Labour_Participation_Rate',
})

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'].str.strip(), format='%d-%m-%Y')

# Extract Month and Year for temporal analysis
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# 3. Exploratory Data Analysis & Visualization

# Visualization 1: Unemployment Rate by State (Bar Plot)
plt.figure(figsize=(12, 6))
state_unemp = df.groupby('State')['Unemployment_Rate'].mean().sort_values(ascending=False).reset_index()
sns.barplot(data=state_unemp, x='Unemployment_Rate', y='State', hue='State', palette='coolwarm', legend=False)
plt.title('Average Unemployment Rate by State/Region in India')
plt.xlabel('Average Estimated Unemployment Rate (%)')
plt.ylabel('State')
plt.tight_layout()
plt.show()

# Visualization 2: Over-time Trends (Impact of COVID-19 around 2020)
plt.figure(figsize=(14, 6))
sns.lineplot(data=df, x='Date', y='Unemployment_Rate', estimator='mean', errorbar=None, color='firebrick')
plt.title('Overall Trend of Unemployment Rate in India (Over Time)')
plt.xlabel('Date')
plt.ylabel('Mean Estimated Unemployment Rate (%)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Heatmap of correlations between numerical variables
plt.figure(figsize=(8, 6))
numeric_df = df[['Unemployment_Rate', 'Employed', 'Labour_Participation_Rate']]
sns.heatmap(numeric_df.corr(), annot=True, cmap='Blues', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
