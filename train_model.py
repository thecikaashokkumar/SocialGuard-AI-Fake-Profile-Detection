import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load datasets
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

# Clean column names
train_data.columns = train_data.columns.str.strip().str.lower()
test_data.columns = test_data.columns.str.strip().str.lower()

target_column = "fake"

X_train = train_data.drop(target_column, axis=1)
y_train = train_data[target_column]

X_test = test_data.drop(target_column, axis=1)
y_test = test_data[target_column]

print("Training features:")
print(X_train.columns)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save BOTH model and feature names
pickle.dump((model, X_train.columns.tolist()), open("model/model.pkl", "wb"))

print("Model saved successfully!")
