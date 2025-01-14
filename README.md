# Cancer-Detection
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
columns = ['ID', 'Diagnosis'] + [f'Feature_{i}' for i in range(1, 31)]
data = pd.read_csv(url, header=None, names=columns)

print("Dataset head:\n", data.head())
print("\nDataset information:\n", data.info())
print("\nSummary statistics:\n", data.describe())

# target variable
data['Diagnosis'] = data['Diagnosis'].map({'M': 1, 'B': 0})  # Malignant=1, Benign=0

# Split the data into features and target
X = data.iloc[:, 2:]  # Exclude ID and Diagnosis
y = data['Diagnosis']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Perform feature selection
selector = SelectKBest(score_func=f_classif, k=10)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# Train multiple models and evaluate
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC(random_state=42)
}

results = {}
for name, model in models.items():
    # Train the model
    model.fit(X_train_selected, y_train)
    # Make predictions
    y_pred = model.predict(X_test_selected)
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"\n{name}:\n")
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Identify the best model
best_model_name = max(results, key=results.get)
print(f"\nBest Model: {best_model_name} with Accuracy: {results[best_model_name]:.2f}")

# Visualize the results
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.show()

# Visualize feature importances for Random Forest (if selected)
if best_model_name == "Random Forest":
    best_model = models[best_model_name]
    feature_importances = best_model.feature_importances_
    top_features = np.argsort(feature_importances)[-10:]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances[top_features], y=np.array(X.columns)[selector.get_support()][top_features])
    plt.title("Top 10 Feature Importances")
    plt.show()
