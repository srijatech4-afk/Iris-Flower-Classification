import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --------------------------------
# Load Dataset
# --------------------------------
df = pd.read_csv("Iris.csv")

# --------------------------------
# Features and Target
# --------------------------------
X = df.drop(["Id", "Species"], axis=1)
y = df["Species"]

# --------------------------------
# Train-Test Split
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------------------
# Create and Train Model
# --------------------------------
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# --------------------------------
# Prediction
# --------------------------------
y_pred = model.predict(X_test)

# --------------------------------
# Accuracy
# --------------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy of the Model:")
print(accuracy)

# --------------------------------
# Confusion Matrix
# --------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()

# --------------------------------
# Classification Report
# --------------------------------
print("\nClassification Report:")
print(classification_report(y_test, y_pred))