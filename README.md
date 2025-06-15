# 🎓 Student Performance Tracker (ML Mini Project)

A machine learning project to predict student performance using academic and personal data.

---

## 🚀 Features

- Predict **Pass/Fail** using Random Forest Classifier
- Predict **Grade** based on previous grades
- Interactive **Streamlit Web App**
- Select student by **ID** to view predictions

---

## 📂 Project Structure

StudentPerformanceTracker/
├── data/student_performance.csv
├── src/student_performance_tracker.py
├── src/student_performance_id_app.py
└── README.md


---

## ⚙ How to Run

### Install Required Packages:

```bash
pip install pandas numpy scikit-learn streamlit

cd src
streamlit run student_performance_id_app.py
