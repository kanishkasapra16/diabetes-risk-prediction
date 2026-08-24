import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Diabetes Risk Prediction Dashboard",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# LOAD TRAINED MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("diabetes_random_forest_model.pkl")


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Model file 'diabetes_random_forest_model.pkl' was not found. "
        "Make sure app.py and the model file are in the same folder."
    )
    st.stop()


# -----------------------------
# TITLE
# -----------------------------
st.title("🩺 AI-Powered Diabetes Risk Prediction")
st.write("Enter patient health information to estimate diabetes risk.")

st.divider()


# -----------------------------
# INPUT SECTION
# -----------------------------
col1, col2 = st.columns(2)

with col1:

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=1,
        step=1
    )

    glucose = st.number_input(
        "Glucose Level",
        min_value=0.0,
        max_value=300.0,
        value=120.0,
        step=1.0
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=1.0
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
        step=1.0
    )


with col2:

    insulin = st.number_input(
        "Insulin Level",
        min_value=0.0,
        max_value=1000.0,
        value=125.0,
        step=1.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        step=0.1
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.50,
        step=0.01
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30,
        step=1
    )


st.divider()


# -----------------------------
# PREDICTION BUTTON
# -----------------------------
if st.button("🔍 Predict Diabetes Risk", use_container_width=True):

    # Create input dataframe
    input_data = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]],
        columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]
    )

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Calculate probability
    prediction_probability = model.predict_proba(input_data)[0][1]

    st.divider()

    # -----------------------------
    # PREDICTION RESULT
    # -----------------------------
    st.header("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Estimated Diabetes Risk",
            f"{prediction_probability * 100:.2f}%"
        )

    with col2:

        if prediction == 1:
            st.error("⚠️ Higher predicted diabetes risk")
        else:
            st.success("✅ Lower predicted diabetes risk")

    # Progress bar
    st.progress(float(prediction_probability))

    st.caption(
        "This tool is for educational and analytical purposes only "
        "and is not a medical diagnosis."
    )

    st.divider()


    # -----------------------------
    # MODEL INSIGHTS
    # -----------------------------
    st.header("📊 Model Insights")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model Accuracy", "74.03%")

    with col2:
        st.metric("Dataset Records", "768")

    with col3:
        st.metric("Input Features", "8")


    # -----------------------------
    # FEATURE IMPORTANCE
    # -----------------------------
    st.subheader("Feature Importance")

    feature_names = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ]

    if hasattr(model, "feature_importances_"):

        feature_importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        })

        feature_importance = feature_importance.sort_values(
            by="Importance",
            ascending=True
        )

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.barh(
            feature_importance["Feature"],
            feature_importance["Importance"]
        )

        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Features")
        ax.set_title("Random Forest Feature Importance")

        st.pyplot(fig)


    st.divider()


    # -----------------------------
    # PATIENT INPUT SUMMARY
    # -----------------------------
    st.header("📋 Patient Input Summary")

    patient_data = pd.DataFrame({
        "Health Parameter": [
            "Pregnancies",
            "Glucose",
            "Blood Pressure",
            "Skin Thickness",
            "Insulin",
            "BMI",
            "Diabetes Pedigree Function",
            "Age"
        ],

        "Entered Value": [
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]
    })

    st.dataframe(
        patient_data,
        use_container_width=True,
        hide_index=True
    )


    # -----------------------------
    # FINAL SUMMARY
    # -----------------------------
    st.subheader("📝 Prediction Summary")

    if prediction == 1:
        st.warning(
            f"The model estimates a diabetes risk probability of "
            f"{prediction_probability * 100:.2f}%. "
            "The prediction falls into the higher-risk category according "
            "to this machine learning model."
        )
    else:
        st.success(
            f"The model estimates a diabetes risk probability of "
            f"{prediction_probability * 100:.2f}%. "
            "The prediction falls into the lower-risk category according "
            "to this machine learning model."
        )

    st.caption(
        "⚠️ Important: This application is a machine learning project "
        "created for educational purposes. It should not be used as a "
        "replacement for professional medical diagnosis or advice."
    )