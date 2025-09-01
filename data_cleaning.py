
import pandas as pd

# Load the dataset from your local path
df = pd.read_csv(r"E:\analytics\telco_churn.csv")

# 1. Drop customerID — it's just an identifier, not useful for analysis
df.drop('customerID', axis=1, inplace=True)

# 2. Convert 'TotalCharges' to numeric — some entries are blank strings
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# 3. Drop rows with missing values (after conversion)
df.dropna(inplace=True)

# 4. Encode binary categorical columns (Yes/No → 1/0)
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# 5. One-hot encode multi-category columns (like InternetService, Contract)
df = pd.get_dummies(df, drop_first=True)

# 6. Save the cleaned dataset (optional but useful)
df.to_csv(r"E:\analytics\cleaned_telco_churn.csv", index=False)

# 7. Print confirmation
print("✅ Data cleaned successfully!")
print("New shape:", df.shape)
