# Welcome to our project!
import pandas as pd
import numpy as np
df = pd.read_csv(r"C:\Users\MCC\Downloads\eye cancer.csv")
def SHAPE(df):
    print("First of all our Data shape is (# of rows, # of columns):")
    print(df.shape)

SHAPE(df)

print('Before we start, if there\'s any step you\'d like to skip, enter "exit" in its field!')

def COUNT(df):
    print("Choose an option:")
    print("1 - Count non-null values in each column")
    print("2 - Get total number of rows in the dataset")
    print("3 - Count non-null values in a specific column")
    print("4 - Count missing (null) values")
    print("5 - Show value counts for a specific column")    
    you = input("Enter the operation here: ")
    if you.lower() == "exit":
        print("Exited count operation.")
        return "Exited"
    if you == "1":
        num_each_col = df.count()
        print("Non-null values in each column:")
        print(num_each_col)
        return num_each_col
    elif you == "2":
        total_rows = df.shape[0]
        print("Total number of rows:")
        print(total_rows)
        return total_rows
    elif you == "3":
        e = input("Enter column name: ")
        if e.lower() == "exit":
            print("Exited count operation.")
            return "Exited"
        valurep = df[e].count()
        print(f"Non-null values in column '{e}': {valurep}")
        return valurep
    elif you == "4":
        print("Choose missing value count option:")
        print("1 - Missing values for all columns")
        print("2 - Missing values for a specific column")
        s = input("Enter your choice: ")
        if s.lower() == "exit":
            print("Exited count operation.")
            return "Exited"
        if s == "1":
            missing_all = df.isnull().sum()
            print("Missing values per column:")
            print(missing_all)
            return missing_all
        elif s == "2":
            d = input("Enter column name: ")
            if d.lower() == "exit":
                print("Exited count operation.")
                return "Exited"
            missing_column = df[d].isnull().sum()
            print(f"Missing values in column '{d}':")
            print(missing_column)
            return missing_column
        else:
            print("Invalid input, please try again.")
            return "Invalid input"
    elif you == "5":
        r = input("Enter column name: ")
        if r.lower() == "exit":
            print("Exited count operation.")
            return "Exited"
        value_counts = df[r].value_counts()
        print(f"Value counts for column '{r}':")
        print(value_counts)
        return value_counts
    else:
        print("Exited.")
        return "Exited"

result = COUNT(df)
print(result)

def DATA_CLEANING(df):
    df['Gender'] = df['Gender'].replace('Other', np.nan)
    df['Gender'] = df['Gender'].fillna(method='ffill')
    df["Age"] = df["Age"].fillna(df["Age"].mean())
    df['Genetic_Markers'] = df['Genetic_Markers'].fillna(method='bfill')
    return "Data cleaning completed."

result = DATA_CLEANING(df)
print(result)

def READING_DATA(df):
    print("1: View all data")
    print("2: Read a specific part of the data (value, rows, columns...)")
    x_local = input("Choose the number of the operation: ")
    if x_local.lower() == "exit":
        print("Exited reading data operation.")
        return "Exited"

    if x_local == "1":
        return df
    elif x_local == "2":
        print("1: Read a specific value")
        print("2: Read a range of rows and columns")
        print("3: Read specific rows and columns")
        print("4: Read certain rows with selected columns")
        y = input("Choose read type: ")
        if y.lower() == "exit":
            print("Exited reading data operation.")
            return "Exited"

        if y == "1":
            a = int(input("Enter the row index: "))
            b = input("Enter the column name: ")
            if b.lower() == "exit":
                print("Exited reading data operation.")
                return "Exited"
            return df.loc[a, b]
        elif y == "2":
            a = int(input("Enter start row index: "))
            b = int(input("Enter end row index: "))
            c = input("Enter start column name: ")
            d = input("Enter end column name: ")
            if c.lower() == "exit" or d.lower() == "exit":
                print("Exited reading data operation.")
                return "Exited"
            return df.loc[a:b, c:d]
        elif y == "3":
            a = list(map(int, input("Enter row indices separated by space: ").split()))
            b = input("Enter column names separated by space: ").split()
            if "exit" in [str(i).lower() for i in a] or "exit" in [col.lower() for col in b]:
                print("Exited reading data operation.")
                return "Exited"
            return df.loc[a, b]
        elif y == "4":
            a = int(input("Enter start row index: "))
            b = int(input("Enter end row index: "))
            c = input("Enter column names separated by space: ").split()
            if "exit" in [col.lower() for col in c]:
                print("Exited reading data operation.")
                return "Exited"
            return df.loc[a:b, c]
        else:
            return "Exited"
    else:
        return "Exited"

result = READING_DATA(df)
print(result)

def TYPE_OF_DATA(df):
    print("Choose an operation:")
    print("1: Show data types of all columns")
    print("2: Show data type of a specific column")
    s = input("Enter the operation you want: ")
    if s.lower() == "exit":
        print("Exited type of data operation.")
        return "Exited"

    if s == "1":
        types = df.dtypes
        print(types)
        return types
    elif s == "2":
        c = input("Enter the column name: ")
        if c.lower() == "exit":
            print("Exited type of data operation.")
            return "Exited"
        dtype = df[c].dtype
        print(dtype)
        return dtype
    else:
        print("Exited.")
        return "Exited"

result = TYPE_OF_DATA(df)
print(result)

def DATA_MODIFY(df):
    print("1: Modify a specific cell value")
    print("2: Add a new row")
    choice = input("Choose an operation: ")
    if choice.lower() == "exit":
        print("Exited data modify operation.")
        return "Exited"
    if choice == "1":
        a = int(input("Enter the row index number to modify: "))
        b = input("Enter the column name: ")
        if b.lower() == "exit":
            print("Exited data modify operation.")
            return "Exited"
        c = input("Enter the new value: ")
        value = df[b].dtype.type(c) 
        df.loc[a, b] = value
        return df.loc[a, b]
    elif choice == "2":
        print("Enter values for the new row:")
        new_values = []
        for col in df.columns:
            val = input(f"{col}: ")
            if val.lower() == "exit":
                print("Exited data modify operation.")
                return "Exited"
            val_converted = df[col].dtype.type(val)
            new_values.append(val_converted)
        df.loc[len(df)] = new_values
        return df.tail(1)
    else:
        return "Exited"

result = DATA_MODIFY(df)
print(result)

def analyze_eye_cancer(data, input_value, input_type='country'):
    if input_type == 'country':  
        df_filtered = data[data['Country'].str.lower() == input_value.lower()]  
        if df_filtered.empty:
            print(f"No data found for country: {input_value}")
            return False
        common_cancer = df_filtered['Cancer_Type'].value_counts().idxmax()       
        group = df_filtered[df_filtered['Cancer_Type'] == common_cancer]['Gender'].value_counts().idxmax()
        avg_age = df_filtered['Age'].mean()
        early_stages = ["Stage I", "Stage II"]
        early_count = df_filtered['Stage_at_Diagnosis'].isin(early_stages).sum()  
        early_ratio = early_count / len(df_filtered)
        early_detected = early_ratio >= 0.5
        common_treatment = df_filtered['Treatment_Type'].value_counts().idxmax()
        print("In", input_value, ":")
        print("- Most common type of eye cancer:", common_cancer)
        print("- Most affected group (by gender):", group)
        print("- Average age at diagnosis:", round(avg_age, 1))
        print("- Early detection achieved?:", "Yes" if early_detected else "No")
        print("- Most commonly used treatment:", common_treatment)
        return True
    
    elif input_type == 'cancer':  
        df_filtered = data[data['Cancer_Type'].str.lower() == input_value.lower()]  
        if df_filtered.empty:
            print(f"No data found for cancer type: {input_value}")
            return False
        common_country = df_filtered['Country'].value_counts().idxmax()
        common_eye = df_filtered['Laterality'].value_counts().idxmax()
        treatment_counts = df_filtered['Treatment_Type'].value_counts()
        top_treatment = treatment_counts.idxmax()
        top_treatment_ratio = treatment_counts.max() / len(df_filtered)
        global_ratio = len(df_filtered) / len(data)
        avg_survival = df_filtered['Survival_Time_Months'].mean()
        avg_age = df_filtered['Age'].mean()
        group = df_filtered['Gender'].value_counts().idxmax()
        print("Cancer Type:", input_value)
        print("- Country with highest occurrence:", common_country)
        print("- Most commonly affected eye:", common_eye)
        print("- Most common treatment:", top_treatment, "(", round(top_treatment_ratio * 100, 1), "% )")
        print("- Global occurrence rate:", round(global_ratio * 100, 2), "%")
        print("- Average survival time:", round(avg_survival, 1), "months")
        print("- Average age at diagnosis:", round(avg_age, 1))
        print("- Most affected group (by gender):", group)
        return True

print("""We will display general information from the dataset based on two categories:
      1 - Cancer
      2 - Country""")

while True:
    w = input("Enter the category name (Cancer or Country) or 'exit' to quit: ").strip().lower()
    if w == "exit":
        print("Exiting the analysis section.")
        break
    elif w == "cancer":
        print("Please choose one of the following cancer types: Retinoblastoma, Melanoma, Lymphoma")
        while True:
            d = input("Enter the cancer type or 'exit' to quit: ").strip()
            if d.lower() == "exit":
                print("Exiting cancer analysis.")
                break
            success = analyze_eye_cancer(df, d, input_type='cancer')
            if success:
                break
            else:
                print("Invalid cancer type. Please try again or enter 'exit' to quit.")
        if d.lower() == "exit":
            continue
        else:
            break
    elif w == "country":
        print("Please enter a valid country name.")
        while True:
            country = input("Enter the country name or 'exit' to quit: ").strip()
            if country.lower() == "exit":
                print("Exiting country analysis.")
                break
            success = analyze_eye_cancer(df, country, input_type='country')
            if success:
                break
            else:
                print("Invalid country name. Please try again or enter 'exit' to quit.")
        if country.lower() == "exit":
            continue
        else:
            break
    else:
        print("Invalid category. Please enter 'Cancer', 'Country', or 'exit'.")

print("Thank you for using the eye cancer data analysis program.")
print()

