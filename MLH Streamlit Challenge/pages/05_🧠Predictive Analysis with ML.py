import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump
from joblib import load
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

IMAGE_MLH = "media/MLH.png"

st.set_page_config(page_title="ShellHacks: MLH Streamlit Challenge", layout="wide",
                   page_icon="🌊", initial_sidebar_state="expanded")

with st.sidebar:
    st.image(IMAGE_MLH, use_column_width=True)
    st.title("ShellHacks: MLH Streamlit Challenge")


def predict_water_quality(df):
    # Select relevant features for prediction
    features = df[['Depth m', 'Temp °C', 'pH', 'ODO mg/L']]
    st.write("**Features used for prediction**: Depth of EXO 2 to Ocean Surface, Temperature in Celsius, and pH Levels "
             "in the Ocean", features.head())

    # Check if model exists
    model_file = 'data.pkl'
    if os.path.exists(model_file):
        model = load(model_file)

        # Make predictions
        predictions = model.predict(features)

        # Display predictions
        st.subheader("Predicted Dissolved Oxygen (ODO mg/L)")
        prediction_df = df.copy()
        prediction_df['Predicted ODO mg/L'] = predictions
        st.dataframe(prediction_df[['Depth m', 'Temp °C', 'pH', 'Predicted ODO mg/L']])

        # Optionally, display MSE (if test data available)
        if 'ODO mg/L' in df.columns:
            true_values = df['ODO mg/L']
            mse = mean_squared_error(true_values, predictions)
            st.write(f"**Mean Squared Error (MSE):** {mse}")
    else:
        st.error(f"Model file {model_file} not found. Train the model first.")


st.title("ML Model Prediction for Water Quality")
st.write("This section ")
uploaded_file = st.file_uploader("Upload a CSV file for prediction", type=["csv"])

if uploaded_file is not None:
    # Load the uploaded file
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully")

    # Run the prediction function
    predict_water_quality(df)

else:
    st.warning("Please upload a CSV file to get predictions.")