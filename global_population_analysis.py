# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 09:20:20 2025

@author: SARITA
"""
#Part-1 Data Cleaning
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
df_pop = pd.read_csv("E:\population_dataset.csv", skiprows=4)
df_country_meta = pd.read_csv("E:\Metadata_Country_API_SP.POP.TOTL_DS2_en_csv_v2_26346.csv")

# Drop unnecessary columns
df_pop.drop(columns=['Indicator Name', 'Indicator Code'], inplace=True, errors='ignore')
df_pop = df_pop.loc[:, ~df_pop.columns.str.contains('^Unnamed')]
df_pop.dropna(subset=df_pop.columns[2:], how='all', inplace=True)

# Fill missing population values
year_columns = [col for col in df_pop.columns if col.isdigit()]
df_pop[year_columns] = df_pop[year_columns].ffill(axis=1)

# Merge with country metadata
df_merged = pd.merge(df_pop, df_country_meta, how='left', on='Country Code')

# Save cleaned data
df_merged.to_csv('Cleaned_Merged_Population_Data.csv', index=False)

#Part-2 Data Visualization

#1. Top 10 Countries by Population in 2022
top_10_2022 = df_merged[['Country Name', '2022']].sort_values(by='2022', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='2022', y='Country Name', data=top_10_2022, palette='viridis')
plt.title('Top 10 Countries by Population (2022)')
plt.xlabel('Population')
plt.ylabel('Country')
plt.tight_layout()
plt.show()

#2. Total Population by Region (2022)
region_2022 = df_merged.groupby('Region')['2022'].sum().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='2022', y='Region', data=region_2022.sort_values('2022'), palette='coolwarm')
plt.title('Total Population by Region (2022)')
plt.xlabel('Population')
plt.ylabel('Region')
plt.tight_layout()
plt.show()

#3. Population Growth for a Specific Country (e.g., India)
india_data = df_merged[df_merged['Country Name'] == 'India']
years = [str(year) for year in range(1960, 2023)]
population_india = india_data[years].values.flatten()

plt.figure(figsize=(12, 6))
plt.plot(years, population_india, marker='o', color='green')
plt.xticks(rotation=45)
plt.title('Population Growth of India (1960–2022)')
plt.xlabel('Year')
plt.ylabel('Population')
plt.grid(True)
plt.tight_layout()
plt.show()

# Exploratory Data Analysis (EDA) and  Statistical Analysis

# Step-by-Step EDA 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv('Cleaned_Merged_Population_Data.csv')

# Check structure and data types
print(df.info())

# Summary statistics for recent years
year_columns = [str(year) for year in range(2000, 2023)]
print("\nSummary Statistics (2000–2022):")
print(df[year_columns].describe())

#1. Missing Values Check
missing = df.isnull().sum()
print("\nMissing values:\n", missing[missing > 0])

#2. Top 10 Countries with Highest Population in 2022
top_10 = df[['Country Name', '2022']].sort_values(by='2022', ascending=False).head(10)
print(top_10)

#3. Population Growth (2000–2022)
df['Growth_2000_2022'] = df['2022'] - df['2000']
top_growth = df[['Country Name', 'Growth_2000_2022']].sort_values(by='Growth_2000_2022', ascending=False).head(10)
print("Top 10 Countries by Population Growth (2000–2022):")
print(top_growth)

#4. Population Distribution by Region (2022)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[df['Region'].notnull()], x='Region', y='2022')
plt.xticks(rotation=45)
plt.title("Population Distribution by Region (2022)")
plt.tight_layout()
plt.show()

# Statistical Analysis

#1. Correlation Between Population Over the Years
corr_matrix = df[year_columns].astype(float).corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, cmap='Blues', annot=False)
plt.title("Correlation Matrix of Population (2000–2022)")
plt.show()

# 2. Descriptive Stats by Income Group (2022)
income_stats = df.groupby('IncomeGroup')['2022'].describe()
print("\nPopulation Statistics by Income Group (2022):")
print(income_stats)

# 3. Skewness and Kurtosis (2022)
from scipy.stats import skew, kurtosis

pop_2022 = df['2022'].dropna()
print(f"Skewness (2022): {skew(pop_2022):.2f}")
print(f"Kurtosis (2022): {kurtosis(pop_2022):.2f}")

#Add Growth Score (Innovation )
# Calculate population growth percentage (2000 to 2022)
df['Growth (%)'] = ((df['2022'] - df['2000']) / df['2000']) * 100
df['Growth (%)'] = df['Growth (%)'].round(2)
df['Growth (%)'] 

# Top 10 countries by growth rate
top_growth = df[['Country Name', 'Growth (%)']].sort_values(by='Growth (%)', ascending=False).head(10)
print("Top 10 Fastest Growing Countries (2000–2022):")
print(top_growth)

#Creative Treemap (Population by Region)
# Treemap using plotly
import plotly.express as px

fig = px.treemap(df[df['Region'].notnull()], 
                 path=['Region', 'Country Name'], 
                 values='2022', 
                 title='Treemap: Population by Region (2022)')
fig.show()

#Interactive-style Filter (Country-wise Analysis)
# Choose a country
selected_country = "India"

# Line plot for selected country
country_data = df[df['Country Name'] == selected_country][year_columns].values.flatten()

plt.figure(figsize=(10, 5))
plt.plot(year_columns, country_data, marker='o')
plt.title(f'Population Trend: {selected_country} (2000–2022)')
plt.xlabel('Year')
plt.ylabel('Population')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Machine Learning
#1. Linear Regression: Predict Future Population
from sklearn.linear_model import LinearRegression
import numpy as np

# Prepare data for a specific country (e.g., India)
country_name = "India"
country_df = df[df['Country Name'] == country_name]

# Use year columns for training
X = np.array([int(year) for year in year_columns]).reshape(-1, 1)
y = country_df[year_columns].values.flatten()

# Train the model
model = LinearRegression()
model.fit(X, y)

# Predict population for future years
future_years = np.array([2023, 2025, 2030]).reshape(-1, 1)
predictions = model.predict(future_years)

# Print predictions
for year, pred in zip(future_years.flatten(), predictions):
    print(f"Predicted population for {country_name} in {year}: {int(pred):,}")

 #2. KMeans Clustering: Group Countries by Population Trends
 from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Use population data from 2000–2022
X = df[year_columns].dropna().astype(float)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to dataframe
df['Cluster'] = -1
df.loc[X.index, 'Cluster'] = clusters

# View a few countries per cluster
print(df[['Country Name', 'Cluster']].groupby('Cluster').head(5))

 3. Decision Tree Classifier: Predict Region or Income Group
 from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Prepare features and target
X = df[year_columns].dropna().astype(float)
y = df.loc[X.index, 'Region']  # Can change to 'IncomeGroup'

# Encode categorical target
y = y.astype('category').cat.codes

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
