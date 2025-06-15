import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("../data/student_performance.csv")
data.columns = data.columns.str.strip()

# Backup original for ID search
data_original = data.copy()

# Prepare data for training
data_model = data.drop('ID', axis=1).copy()

# Encoding
le_internet = LabelEncoder()
le_extra = LabelEncoder()
le_target = LabelEncoder()

data_model['Internet'] = le_internet.fit_transform(data_model['Internet'])
data_model['Extra_Curricular'] = le_extra.fit_transform(data_model['Extra_Curricular'])
data_model['Target'] = le_target.fit_transform(data_model['Target'])

X = data_model.drop('Target', axis=1)
y = data_model['Target']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# ---------------------------------------------------
# Grade Calculation Function
def calculate_grade(previous_grades):
    if previous_grades >= 90:
        return 'A'
    elif previous_grades >= 80:
        return 'B'
    elif previous_grades >= 70:
        return 'C'
    elif previous_grades >= 60:
        return 'D'
    else:
        return 'F'

# ---------------------------------------------------
# Streamlit App

st.title("🎓 Student Performance Tracker (By Student ID)")
st.write("Select a student ID to predict their performance and grade:")

# Get list of IDs
id_list = data_original['ID'].tolist()

# ID selection dropdown
selected_id = st.selectbox("Select Student ID", id_list)

if selected_id:
    # Get student row
    student_row = data_original[data_original['ID'] == selected_id]
    
    st.subheader("📊 Student Data:")
    st.write(student_row)

    # Prepare for model prediction
    student_row_model = student_row.drop('ID', axis=1).copy()
    student_row_model['Internet'] = le_internet.transform(student_row_model['Internet'])
    student_row_model['Extra_Curricular'] = le_extra.transform(student_row_model['Extra_Curricular'])
    
    student_X = student_row_model.drop('Target', axis=1)
    student_X_scaled = scaler.transform(student_X)

    # Predict
    prediction = model.predict(student_X_scaled)
    result = "Pass" if prediction[0] == 1 else "Fail"

    # Calculate grade
    previous_grades = student_row['Previous_Grades'].values[0]
    grade = calculate_grade(previous_grades)

    st.subheader("🎯 Prediction Result:")
    st.success(f"Predicted Performance: {result}")
    st.info(f"Predicted Grade: {grade}")
