import pandas as pd
import numpy as np
import joblib as jb
import streamlit as st

from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Diabetes Prediction", layout="wide")


@st.cache_resource
def load_resources():
    # load model and scaler once
    model = jb.load('diabetes_model.pkl')
    scaler = jb.load('scaler.pkl')
    return model, scaler


model, scaler = load_resources()

st.title("Diabetes Prediction App")

st.header("Patient input")

# arrange inputs into two columns for compact UI
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=250, value=120)
    bp = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)

with col2:
    skin = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=20.0, format="%.1f")
    insulin = st.number_input("Insulin Level", min_value=0.0, max_value=900.0, value=79.0, format="%.1f")
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")

with col3:
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=5.0, value=0.5, format="%.3f")
    age = st.number_input("Age", min_value=0, max_value=120, value=28)

# feature engineering as used during training
insulin_safe = float(insulin)
log_insulin = np.log(insulin_safe + 1e-6)

data = [pregnancies, glucose, bp, skin, log_insulin, bmi, dpf, age,
        age * pregnancies, bmi * glucose, glucose / (bmi + 1e-6)]


def predict_diabetes(row):
    """Return dict with label and probability."""
    features = pd.DataFrame([row], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                                           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
                                           'Age_Pregnancies', 'BMI_Glucose', 'Glucose_per_BMI'])
    scaled = scaler.transform(features)
    # handle models without predict_proba
    try:
        prob = float(model.predict_proba(scaled)[:, 1][0])
    except Exception:
        # fall back to predict then map to 0/1 probability
        pred = int(model.predict(scaled)[0])
        prob = 0.95 if pred == 1 else 0.05
    label = "Diabetic" if prob >= 0.5 else "Non-Diabetic"
    return {"label": label, "probability": prob}


def format_probability(p):
    return f"{p*100:.1f}%"


# validation
if glucose <= 0:
    st.warning("Glucose should be > 0 for a meaningful prediction.")

if st.button("Predict"):
    result = predict_diabetes(data)
    prob = result['probability']
    label = result['label']

    # top-level display
    left, right = st.columns([2, 3])
    with left:
        if label == "Diabetic":
            st.error(f"Prediction: {label}")
        else:
            st.success(f"Prediction: {label}")
        st.metric("Probability of diabetes", format_probability(prob))
        # progress bar as a visual indicator
        st.progress(min(max(prob, 0.0), 1.0))

    with right:
        st.subheader("Details")
        st.write(f"Model confidence: **{format_probability(prob)}**")
        if prob >= 0.75:
            st.write("This result indicates a high likelihood; please consult a healthcare professional.")
        elif prob >= 0.5:
            st.write("Moderate likelihood — consider further tests and consultation.")
        else:
            st.write("Low likelihood, but if you have concerns, consult a healthcare professional.")

    # create full features (used by model) and a display-friendly view with original inputs only
    full_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
                    'Age_Pregnancies', 'BMI_Glucose', 'Glucose_per_BMI']
    input_df_full = pd.DataFrame([data], columns=full_columns)

    # display/download only the original 8 input columns
    display_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    display_df = input_df_full[display_columns]

    # append model outcomes to the downloadable CSV
    display_df = display_df.copy()
    display_df['Prediction'] = label
    display_df['Probability'] = prob
    display_df['Insulin'] = np.exp(display_df['Insulin']) 

    csv = display_df.to_csv(index=False)
    st.download_button("Download input as CSV", csv, file_name="diabetes_input.csv", mime="text/csv")

    # show raw inputs in an expander — show only the original inputs
    with st.expander("Show raw input values"):
        st.write(display_df)
