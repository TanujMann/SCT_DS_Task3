import pandas as pd
import zipfile
import requests
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 1. Download and extract the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
response = requests.get(url)
with open("bank-additional.zip", "wb") as f:
    f.write(response.content)
with zipfile.ZipFile("bank-additional.zip") as zip_file:
    with zip_file.open('bank-additional/bank-additional-full.csv') as file:
        df = pd.read_csv(file, sep=';')

# 2. Preprocess the data
def preprocess_data(df):
    categorical_cols = df.select_dtypes(include=['object']).columns.drop('y')
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    df['y'] = df['y'].map({'yes': 1, 'no': 0})
    return df

df_processed = preprocess_data(df.copy())

# 3. Split into features and target
X = df_processed.drop('y', axis=1)
y = df_processed['y']

# 4. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Initialize and train the Decision Tree
dt_classifier = DecisionTreeClassifier(
    max_depth=5, 
    min_samples_split=10, 
    random_state=42
)
dt_classifier.fit(X_train, y_train)

# 6. Generate predictions
y_pred = dt_classifier.predict(X_test)

# 7. Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 8. Visualize the Decision Tree with a visible title
plt.figure(figsize=(15, 8))
plot_tree(
    dt_classifier,
    feature_names=X.columns,
    class_names=['No', 'Yes'],
    filled=True,
    fontsize=7
)
ax = plt.gca()
ax.set_title("Customer Purchase Prediction - Decision Tree Visualization", fontsize=18, pad=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust to make sure title is visible
plt.show()
