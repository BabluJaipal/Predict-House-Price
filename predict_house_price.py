# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Step 2: Set the full path to your train.csv
# Change this path according to where you downloaded the file
csv_path = r"C:\Users\Bablu Jaipal\Desktop\Broshkies_Hub\slab_1\train.csv"

# Step 3: Load Dataset
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"The file was not found at: {csv_path}")

df = pd.read_csv(csv_path)

# Step 4: Quick Overview
print("Dataset Shape:", df.shape)
print(df.head())
print(df.info())

# Step 5: Drop irrelevant columns
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

# Step 6: Handle missing values
# Drop columns with >30% missing values
missing_threshold = 0.3
df = df.dropna(axis=1, thresh=int((1 - missing_threshold) * len(df)))

# Fill remaining missing values
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# Step 7: Encode categorical variables (One-Hot Encoding)
df = pd.get_dummies(df, drop_first=True)

# Step 8: Define Features and Target
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# Step 9: Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 10: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Step 11: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)