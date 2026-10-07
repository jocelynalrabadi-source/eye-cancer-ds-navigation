#trained model using supervised learning
print('Its time to train our model using supervised learning')
x=input('Type Ready then enter to see our trained model statistics!')
print()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, Binarizer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, Binarizer
df = pd.read_csv(r"C:\Users\MCC\Downloads\eye cancer.csv")
def DATA_CLEANING(df):
    df['Gender'] = df['Gender'].replace('Other', np.nan)
    df['Gender'] = df['Gender'].fillna(method='ffill')
    df["Age"] = df["Age"].fillna(df["Age"].mean())
    df['Genetic_Markers'] = df['Genetic_Markers'].fillna(method='bfill')
    return "Data cleaning completed."
result = DATA_CLEANING(df)
df['Age_Binary'] = Binarizer(threshold=40).fit_transform(df[['Age']])
df['Radiation_Binary'] = Binarizer(threshold=30).fit_transform(df[['Radiation_Therapy']])
df['Chemo_Binary'] = np.where(df['Chemotherapy'] > 5, 1, 0)
df['Survival_Binary'] = np.where(df['Survival_Time_Months'] > 12, 1, 0)
df.to_csv("binarized_data.csv", index=False)
df['y_true'] = df['Outcome_Status'].apply(lambda x: 1 if x == 'In Remission' else 0)
X = df.drop(columns=['Outcome_Status', 'y_true'], errors='ignore')
X = pd.get_dummies(X)
y = df['y_true']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)
model = GaussianNB()
model.fit(X_train_scaled, y_train_smote)
y_pred = model.predict(X_test_scaled)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not In Remission', 'In Remission']))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
print(f"Recall: {recall_score(y_test, y_pred):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")
print("\nHere's a visualization of our confusion matrix:")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not In Remission', 'In Remission'],
            yticklabels=['Not In Remission', 'In Remission'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
print("\nOur model did a pretty good job!")
