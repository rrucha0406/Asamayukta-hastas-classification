import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# load dataset (use validation or test CSV ideally)
df = pd.read_csv(r"notebooks\mudra_landmarks_test.csv")

# split features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# load label encoder and model
model = joblib.load(r"models/mudra_model.pkl")
le = joblib.load(r"models/label_encoder.pkl")

# encode labels
y_encoded = le.transform(y)

# predict
y_pred = model.predict(X)

# accuracy
acc = accuracy_score(y_encoded, y_pred)
print("Accuracy:", acc)

# detailed report
print("\nClassification Report:\n")
print(classification_report(y_encoded, y_pred))

# confusion matrix
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_encoded, y_pred))
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_encoded, y_pred)
plt.savefig("confusion_matrix.png", bbox_inches="tight")
plt.show()