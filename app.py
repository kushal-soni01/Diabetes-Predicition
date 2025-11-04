import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import joblib as jb
import streamlit as st

model = jb.load('diabetes_model.pkl')
scaler = jb.load('scaler.pkl')

st.title("Diabetes Prediction App")

def predict_diabetes(row):
    features = pd.DataFrame([row], columns=['Pregnancies','Glucose','BloodPressure','SkinThickness',
                                               'Insulin','BMI','DiabetesPedigreeFunction','Age',
                                               'Age_Pregnancies','BMI_Glucose','Glucose_per_BMI'])
    scaled_features = scaler.transform(features)
    prob = model.predict_proba(scaled_features)[:, 1]
    prediction = model.predict(scaled_features)
    return "Diabetic" if prediction[0]==1 else "Non-Diabetic"

pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose Level", min_value=0, max_value=250, value=120)
bp = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=20.0)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=79)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input("Age", min_value=0, max_value=100, value=28)

data = [pregnancies, glucose, bp, skin, np.log(insulin+1e-6), bmi, dpf, age,
        age*pregnancies, bmi*glucose, glucose/(bmi +1e-6)]


if st.button("Predict"):
    result = predict_diabetes(data)
    st.write(f"The prediction is: **{result}**")