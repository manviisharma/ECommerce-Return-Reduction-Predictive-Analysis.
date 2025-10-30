import pandas as pd
import numpy as np
import os

# --- Configuration ---
# 1. Define the filename. **Make sure this matches your CSV file name!**
FILE_NAME = 'ecommerce_returns_synthetic_data.csv'

# --- 1. Data Loading ---
print(f"--- Attempting to load file: {'ecommerce_returns_synthetic_data.csv'} ---")
try:
    # Load the CSV file into a DataFrame named 'df'
    df = pd.read_csv('ecommerce_returns_synthetic_data.csv')
    print(f"SUCCESS: Loaded {len(df)} records.")
except FileNotFoundError:
    print(f"ERROR: The file '{'ecommerce_returns_synthetic_data.csv'}' was not found.")
    print(f"Current working directory: {os.getcwd()}")
    print("Please ensure the CSV file is in the same folder as this script.")
    exit()

# --- 2. Initial Inspection and Cleaning ---

print("\n--- Initial DataFrame Info ---")
df.info()

# 2.1 Convert Date Columns (Crucial First Step)
print("\nConverting Order_Date and Return_Date to datetime objects...")
df['Order_Date'] = pd.to_datetime(df['Order_Date'])
df['Return_Date'] = pd.to_datetime(df['Return_Date'])

# 2.2 Create the Binary Target Variable ('is_returned')
# This is required for predictive modeling (1 = Returned, 0 = Not Returned)
df['is_returned'] = df['Return_Status'].apply(lambda x: 1 if x == 'Returned' else 0)

# --- 3. Feature Engineering ---

# 3.1 Time-based Features for Trend Analysis
print("Engineering time-based features...")
df['Order_Month'] = df['Order_Date'].dt.to_period('M') # For monthly analysis
df['Order_Day_of_Week'] = df['Order_Date'].dt.day_name() # For weekly pattern analysis

# 3.2 Impute Missing Marketing Channel (Necessary for Project Objective)
# Since your data lacks a marketing channel, we simulate one.
# This allows you to complete the 'marketing channel' objective of the project.
print("Synthesizing 'Marketing_Channel' feature...")
channels = ['Organic', 'Paid Search', 'Social Media', 'Email Campaign', 'Direct']
# Assign channels randomly with different probabilities
df['Marketing_Channel'] = np.random.choice(channels, size=len(df), 
                                          p=[0.35, 0.25, 0.2, 0.1, 0.1])


# --- 4. Verification of Prepared Data ---

print("\n--- Verification of Prepared DataFrame ---")
df.info() # Check for datetime conversion and new columns
print("\n--- Quick Look at New Features ---")
print(df[['Order_Date', 'Return_Date', 'is_returned', 'Order_Month', 'Marketing_Channel']].head())

# The DataFrame 'df' is now clean and ready for in-depth EDA and modeling!
# Next, you will perform SQL-style aggregations using Pandas.
# Calculate the overall return rate
total_returns = df['is_returned'].sum()
total_orders = len(df)
overall_return_rate = (total_returns / total_orders) * 100

print(f"\nOverall Return Rate: {overall_return_rate:.2f}%")
# Group by category and calculate return rate
category_analysis = df.groupby('Product_Category').agg(
    Total_Orders=('Order_ID', 'count'),
    Total_Returns=('is_returned', 'sum')
).reset_index()

category_analysis['Return_Rate'] = (category_analysis['Total_Returns'] / category_analysis['Total_Orders']) * 100

print("\n--- Return Rate by Product Category ---")
print(category_analysis.sort_values(by='Return_Rate', ascending=False))
# Group by user location and calculate return rate
location_analysis = df.groupby('User_Location').agg(
    Total_Orders=('Order_ID', 'count'),
    Total_Returns=('is_returned', 'sum')
).reset_index()

location_analysis['Return_Rate'] = (location_analysis['Total_Returns'] / location_analysis['Total_Orders']) * 100

print("\n--- Return Rate by User Location (Top 10) ---")
# Filter out locations with too few orders to ensure reliable rate calculation (optional)
location_analysis = location_analysis[location_analysis['Total_Orders'] > 50]
print(location_analysis.sort_values(by='Return_Rate', ascending=False).head(10))
# Group by marketing channel and calculate return rate
channel_analysis = df.groupby('Marketing_Channel').agg(
    Total_Orders=('Order_ID', 'count'),
    Total_Returns=('is_returned', 'sum')
).reset_index()

channel_analysis['Return_Rate'] = (channel_analysis['Total_Returns'] / channel_analysis['Total_Orders']) * 100

print("\n--- Return Rate by Marketing Channel ---")
print(channel_analysis.sort_values(by='Return_Rate', ascending=False))
# Filter for only returned items and count the reasons
reason_counts = df[df['is_returned'] == 1]['Return_Reason'].value_counts().reset_index()
reason_counts.columns = ['Return_Reason', 'Count']

print("\n--- Top Return Reasons ---")
print(reason_counts)
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# Example Visualization: Return Rate by Category
plt.figure(figsize=(10, 6))
sns.barplot(x='Return_Rate', y='Product_Category', hue='Product_Category', data=category_analysis.sort_values('Return_Rate', ascending=False), palette="viridis", legend=False)
plt.title('Return Rate by Product Category')
plt.xlabel('Return Rate (%)')
plt.ylabel('Product Category')
plt.tight_layout()
plt.savefig('return_rate_by_category.png')
print("Plot saved as 'return_rate_by_category.png'")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier # A robust model choice

print("\n--- 6.1 Defining Features and Target ---")

# Features (X) available at the time of order
feature_cols = [
    'Product_Category', 'Product_Price', 'Order_Quantity',
    'User_Age', 'User_Gender', 'User_Location',
    'Payment_Method', 'Shipping_Method', 'Discount_Applied',
    'Marketing_Channel' # The simulated column
]

# Target variable (y)
target_col = 'is_returned'

X = df[feature_cols]
y = df[target_col]

# Identify categorical features for encoding
categorical_features = X.select_dtypes(include=['object']).columns
print(f"Categorical Features to be encoded: {list(categorical_features)}")
print("\n--- 6.2 Encoding Categorical Data ---")

# Apply One-Hot Encoding
X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)

print(f"Total features after encoding: {X_encoded.shape[1]}")
# Split the data into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")
print("\n--- 6.4 Training Random Forest Classifier ---")

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
# Use 'balanced' to help with the class imbalance (fewer returns than non-returns)
model.fit(X_train, y_train)

print("Model training complete.")
from sklearn.metrics import classification_report, confusion_matrix

print("\n--- 6.5 Model Evaluation on Test Set ---")

# Predict on the test set
y_pred = model.predict(X_test)

# Print the classification report
print("Classification Report:")
# Focus on class 1 (Returned) metrics: Precision, Recall, F1-Score
print(classification_report(y_test, y_pred))

# Optional: Print the Confusion Matrix
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))
# 1. Predict probabilities for the entire encoded dataset
# model.predict_proba() returns probabilities for [Class 0, Class 1]
probabilities = model.predict_proba(X_encoded)

# Get the probability of return (Class 1)
df['return_probability'] = probabilities[:, 1]

# 2. Define the threshold and identify high-risk products
# A common threshold for high-risk could be 60% (0.6) or higher
HIGH_RISK_THRESHOLD = 0.60

high_risk_products_df = df[df['return_probability'] >= HIGH_RISK_THRESHOLD]

# 3. Select final columns and export
final_columns = [
    'Order_ID', 'Product_ID', 'Product_Category', 'Product_Price',
    'User_Location', 'Return_Reason', 'return_probability'
]

high_risk_products_export = high_risk_products_df[final_columns].sort_values(
    by='return_probability', ascending=False
)

# Export the final CSV deliverable
CSV_FILE_NAME = 'high_risk_products.csv'
high_risk_products_export.to_csv(CSV_FILE_NAME, index=False)

print(f"\n✅ High-risk products identified and exported.")
print(f"Total High-Risk Products (Prob >= {HIGH_RISK_THRESHOLD*100}%): {len(high_risk_products_export)}")
print(f"CSV file saved as: {CSV_FILE_NAME}")
# --- EXPORT 1: The Full Data for Power BI ---
# Add this line near the end of your script
df.to_csv('analysis_data_for_powerbi.csv', index=False)
# --- EXPORT 2: The High-Risk Deliverable ---
CSV_FILE_NAME = 'high_risk_products.csv' 
# The DataFrame was defined in Step 6.6 as:
# high_risk_products_export = high_risk_products_df[final_columns].sort_values(...)

high_risk_products_export.to_csv(CSV_FILE_NAME, index=False)
