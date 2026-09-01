import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('cosmetics_sales_data.csv')

# a) Check for missing values
print("Missing values per column:\n", df.isnull().sum())

# b) Feature engineering
# Price per box
df['Price_per_box'] = df['Amount ($)'] / df['Boxes Shipped']

# Extract Month from Date
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month

# Create target variable: Revenue_Class (High/Low based on median Amount)
median_amount = df['Amount ($)'].median()
df['Revenue_Class'] = np.where(df['Amount ($)'] > median_amount, 'High', 'Low')

print("\nMedian Amount used as threshold:", median_amount)
print(df[['Amount ($)', 'Revenue_Class']].head())

# c) Feature selection - drop columns not used as predictors
df_model = df.drop(columns=['Sales Person', 'Date', 'Amount ($)'])

# d) Encode categorical variables (Country, Product)
df_encoded = pd.get_dummies(df_model, columns=['Country', 'Product'], drop_first=True)

# Encode target variable: High = 1, Low = 0
df_encoded['Revenue_Class'] = np.where(df_encoded['Revenue_Class'] == 'High', 1, 0)

print("\nFinal processed dataset shape:", df_encoded.shape)
print(df_encoded.head())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Separate features (X) and target (y)
X = df_encoded.drop(columns=['Revenue_Class'])
y = df_encoded['Revenue_Class']

# Split into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

print("\nModel trained successfully.")
print("Model coefficients:\n", model.coef_)



from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, roc_auc_score, RocCurveDisplay
)
import matplotlib.pyplot as plt

# a) Cross-validation (5-fold) on training data
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print("Cross-validation accuracy scores:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())
print("Std deviation:", cv_scores.std())

# b) Predict on the test set
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # probability of "High" class

# Accuracy
test_accuracy = accuracy_score(y_test, y_pred)
print("\nTest set accuracy:", test_accuracy)

# Classification report (precision, recall, F1)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Low', 'High']))

# Confusion matrix (visualization)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low', 'High'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - Logistic Regression')
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# ROC curve (visualization)
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc_score = roc_auc_score(y_test, y_proba)
print("\nROC-AUC Score:", auc_score)

RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title('ROC Curve - Logistic Regression')
plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
plt.show()