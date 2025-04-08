import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


file_path = r"C:\Users\arkak\Desktop\stock_hmm_output.csv"
# Read the CSV with a header and set the first column as the index
# (the first column in your file is empty and acts as an index)
data = pd.read_csv(file_path, index_col=0)


# Optional: Display the first few rows to verify the data
print(data.head())

# Define the four features and the target variable
features = ["Return", "Volatility", "HighR", "LowR"]
target = "State"
X = data[features]
y = data[target]

# Standardize the features for better SVM performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into 70% training and 30% testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Initialize the SVM classifier; using a linear kernel here (you can change it as needed)
svm_model = SVC(kernel='linear', random_state=42)

# Train the SVM model using the training data
svm_model.fit(X_train, y_train)

# Predict the target variable on the test set
y_pred = svm_model.predict(X_test)

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of SVM Model: {accuracy:.2f}\n")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
