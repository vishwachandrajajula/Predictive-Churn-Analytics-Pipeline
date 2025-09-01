import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the cleaned dataset
df = pd.read_csv(r"E:\analytics\cleaned_telco_churn.csv")

# 1. Separate features and target
X = df.drop('Churn', axis=1)  # All input features
y = df['Churn']               # Target label (0 or 1)

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Confirm shapes
print("✅ Data split and scaled successfully!")
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)
