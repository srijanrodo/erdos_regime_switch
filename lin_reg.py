import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

file_path = r"C:\Users\arkak\Desktop\stock_hmm_output.csv"
# Read the CSV with a header and set the first column as the index
# (the first column in your file is empty and acts as an index)
data = pd.read_csv(file_path, index_col=0)

# Check that the columns match expected ones:
# Expected columns: Return, Volatility, HighR, LowR, State, Ticker
print(data.head())

# Select the four feature columns and the target variable
features = ["Return", "Volatility", "HighR", "LowR"]
target = "State"
X = data[features]
y = data[target]

# Standardize features for better performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into 70% training and 30% testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance with accuracy, confusion matrix, and classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of Logistic Regression Model: {accuracy:.2f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
