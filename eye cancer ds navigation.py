import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, Binarizer
from imblearn.over_sampling import SMOTE
df = pd.read_csv(r"C:\Users\MCC\Downloads\eye cancer.csv")
conn = sqlite3.connect("data_operations_log.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS operations_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        operation TEXT,
        detail TEXT
    )
''')
conn.commit()

def log_operation(operation, detail=""):
    cursor.execute(
        "INSERT INTO operations_log (timestamp, operation, detail) VALUES (?, ?, ?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), operation, detail)
    )
    conn.commit()

st.title("Eye Cancer Data Analysis")
st.subheader("Dataset Shape")
st.write("Our Data shape is (rows, columns):", df.shape)

with st.expander("Count Operations"):
    count_option = st.selectbox("Choose an operation:", [
        "Select one", "Count non-null values in each column", "Total number of rows",
        "Count non-null values in a specific column", "Count missing values",
        "Show value counts for a specific column"
    ])
    if count_option == "Count non-null values in each column":
        st.write(df.count())
        log_operation("Count non-null", "All columns")
    elif count_option == "Total number of rows":
        st.write(df.shape[0])
        log_operation("Total rows")
    elif count_option == "Count non-null values in a specific column":
        col = st.selectbox("Select column:", df.columns)
        st.write(df[col].count())
        log_operation("Count non-null", f"Column: {col}")
    elif count_option == "Count missing values":
        missing_type = st.radio("Choose:", ["All columns", "Specific column"])
        if missing_type == "All columns":
            st.write(df.isnull().sum())
            log_operation("Missing values", "All columns")
        else:
            col = st.selectbox("Select column:", df.columns)
            st.write(df[col].isnull().sum())
            log_operation("Missing values", f"Column: {col}")
    elif count_option == "Show value counts for a specific column":
        col = st.selectbox("Select column:", df.columns)
        st.write(df[col].value_counts())
        log_operation("Value counts", f"Column: {col}")
def DATA_CLEANING(df):
    df['Gender'] = df['Gender'].replace('Other', np.nan)
    df['Gender'] = df['Gender'].fillna(method='ffill')
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Genetic_Markers'] = df['Genetic_Markers'].fillna(method='bfill')
    return df
df = DATA_CLEANING(df)
st.success("Data cleaned successfully.")

data_view_option = st.selectbox("View or slice the data:", ["Full Data", "Slice Data"])
if data_view_option == "Full Data":
    st.dataframe(df)
    log_operation("View", "Full dataset")
elif data_view_option == "Slice Data":
    row_start = st.number_input("Start row", min_value=0, max_value=len(df)-1, value=0)
    row_end = st.number_input("End row", min_value=0, max_value=len(df)-1, value=5)
    columns = st.multiselect("Columns to display", df.columns.tolist(), default=df.columns.tolist())
    st.dataframe(df.loc[row_start:row_end, columns])
    log_operation("Slice Data", f"Rows: {row_start}-{row_end}, Columns: {columns}")

with st.expander("Data Types"):
    dtype_option = st.radio("Choose:", ["All columns", "Specific column"])
    if dtype_option == "All columns":
        st.write(df.dtypes)
        log_operation("Data types", "All columns")
    else:
        col = st.selectbox("Select column:", df.columns)
        st.write(f"{col}: {df[col].dtype}")
        log_operation("Data type", f"Column: {col}")

with st.expander("Modify Data"):
    modify_option = st.radio("Choose:", ["Modify cell", "Add new row"])
    if modify_option == "Modify cell":
        row = st.number_input("Row index", min_value=0, max_value=len(df)-1)
        col = st.selectbox("Column:", df.columns)
        new_val = st.text_input("New value:")
        if st.button("Apply Change"):
            df.at[row, col] = df[col].dtype.type(new_val)
            st.success("Value updated.")
            log_operation("Modify Cell", f"Row: {row}, Column: {col}, New Value: {new_val}")
    elif modify_option == "Add new row":
        inputs = {}
        for col in df.columns:
            inputs[col] = st.text_input(f"{col}")
        if st.button("Add Row"):
            new_row = [df[col].dtype.type(inputs[col]) for col in df.columns]
            df.loc[len(df)] = new_row
            st.success("Row added.")
            log_operation("Add Row", f"Values: {inputs}")
def analyze_eye_cancer(data, value, mode='country'):
    if mode == 'country':
        filtered = data[data['Country'].str.lower() == value.lower()]
        if filtered.empty:
            return "No data for this country."
        return {
            'Most Common Cancer': filtered['Cancer_Type'].value_counts().idxmax(),
            'Most Affected Gender': filtered['Gender'].value_counts().idxmax(),
            'Avg Age': round(filtered['Age'].mean(), 1),
            'Early Detection': (filtered['Stage_at_Diagnosis'].isin(["Stage I", "Stage II"]).sum() / len(filtered)) >= 0.5,
            'Common Treatment': filtered['Treatment_Type'].value_counts().idxmax()
        }
    else:
        filtered = data[data['Cancer_Type'].str.lower() == value.lower()]
        if filtered.empty:
            return "No data for this cancer type."
        return {
            'Top Country': filtered['Country'].value_counts().idxmax(),
            'Commonly Affected Eye': filtered['Laterality'].value_counts().idxmax(),
            'Common Treatment': filtered['Treatment_Type'].value_counts().idxmax(),
            'Treatment Ratio (%)': round(filtered['Treatment_Type'].value_counts().max() / len(filtered) * 100, 1),
            'Global Occurrence Rate (%)': round(len(filtered)/len(data)*100, 2),
            'Avg Survival Time (months)': round(filtered['Survival_Time_Months'].mean(), 1),
            'Avg Age': round(filtered['Age'].mean(), 1),
            'Most Affected Gender': filtered['Gender'].value_counts().idxmax()
        }
st.subheader("Analyze by Category")
category = st.radio("Choose category:", ["Country", "Cancer"])
if category == "Country":
    selected_country = st.text_input("Enter country name")
    if st.button("Analyze Country") and selected_country:
        result = analyze_eye_cancer(df, selected_country, 'country')
        if isinstance(result, dict):
            st.json(result)
            log_operation("Analyze Country", selected_country)
        else:
            st.warning(result)
if category == "Cancer":
    selected_cancer = st.selectbox("Choose cancer type", ["Retinoblastoma", "Melanoma", "Lymphoma"])
    if st.button("Analyze Cancer"):
        result = analyze_eye_cancer(df, selected_cancer, 'cancer')
        if isinstance(result, dict):
            st.json(result)
            log_operation("Analyze Cancer", selected_cancer)
        else:
            st.warning(result)
st.info("Thank you for using the eye cancer data analysis program.")
with st.expander("View Operations Log Database"):
    cursor.execute("SELECT * FROM operations_log ORDER BY timestamp DESC")
    logs = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    if logs:
        df_logs = pd.DataFrame(logs, columns=columns)
        st.dataframe(df_logs)
    else:
        st.write("No operations logged yet.")
import sqlite3
import pandas as pd
conn = sqlite3.connect("data_operations_log.db")
df_log = pd.read_sql_query("SELECT * FROM operations_log", conn)
print(df_log)
conn.close()
df['Age_Binary'] = Binarizer(threshold=40).fit_transform(df[['Age']])
df['Radiation_Binary'] = Binarizer(threshold=30).fit_transform(df[['Radiation_Therapy']])
df['Chemo_Binary'] = np.where(df['Chemotherapy'] > 5, 1, 0)
df['Survival_Binary'] = np.where(df['Survival_Time_Months'] > 12, 1, 0)
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
st.subheader("Model Evaluation Metrics")
st.write("### Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Not In Remission', 'In Remission'],
            yticklabels=['Not In Remission', 'In Remission'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
st.pyplot(fig)
st.write("### Classification Report")
st.text(classification_report(y_test, y_pred, target_names=['Not In Remission', 'In Remission']))
st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
st.write(f"**Precision:** {precision_score(y_test, y_pred):.2f}")
st.write(f"**Recall:** {recall_score(y_test, y_pred):.2f}")
st.write(f"**F1 Score:** {f1_score(y_test, y_pred):.2f}")
st.success("Model training and evaluation complete! Our model did a pretty good job.")

