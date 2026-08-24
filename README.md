# 🩺 AI-Powered Diabetes Risk Prediction

An interactive machine learning web application that predicts the estimated risk of diabetes based on patient health parameters. The application uses a **Random Forest Machine Learning model** and is deployed using **Streamlit**.

## 🚀 Live Demo

🔗 LIVE_APP_URL

## 📌 Project Overview

This project uses machine learning to analyze patient health information and estimate the probability of diabetes risk. Users can enter health-related parameters through an interactive web interface and receive an instant prediction.

The project is developed for educational and analytical purposes and is not intended to replace professional medical diagnosis.

## ✨ Features

- Interactive and user-friendly Streamlit dashboard
- Diabetes risk prediction using a trained Random Forest model
- Probability-based risk estimation
- Low-risk and higher-risk prediction categories
- Patient input summary
- Model accuracy display
- Dataset information
- Feature importance visualization
- Educational disclaimer

## 📊 Input Features

The model uses the following 8 health parameters:

1. Pregnancies
2. Glucose
3. Blood Pressure
4. Skin Thickness
5. Insulin
6. BMI
7. Diabetes Pedigree Function
8. Age

## 🤖 Machine Learning Model

The project uses a **Random Forest Classifier** for diabetes risk prediction.

The trained model is saved as:

`diabetes_random_forest_model.pkl`

## 📈 Model Performance

- **Model:** Random Forest Classifier
- **Dataset Records:** 768
- **Input Features:** 8
- **Model Accuracy:** 74.03%

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- Joblib
- Matplotlib

## 📁 Project Structure

```text
diabetes-risk-prediction/
│
├── app.py
├── diabetes_random_forest_model.pkl
├── requirements.txt
└── README.md
