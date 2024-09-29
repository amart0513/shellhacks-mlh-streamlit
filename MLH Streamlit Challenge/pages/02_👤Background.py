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
IMAGE1 = "media/BBC_Research1.jpg"
IMAGE2 = "media/BBC_Research2.jpg"
IMAGE_SH = "media/SH.png"


st.set_page_config(page_title="ShellHacks: MLH Streamlit Challenge", layout="wide",
                   page_icon="🌊", initial_sidebar_state="expanded")

with st.sidebar:
    st.image(IMAGE_MLH, use_column_width=True)
    st.title("ShellHacks: MLH Streamlit Challenge")



st.title("Background")
st.subheader("About our Project")
col1, col2 = st.columns(2)

with col1:
    st.write("Our project focused on real-time water quality analysis using machine learning, with data collected from Biscayne Bay and \
               Haulover Beach. By integrating field data directly from these locations into our predictive models, we ensured that the machine \
               learning algorithms were trained on high-quality, location-specific data. This allowed for the creation of an accurate predictor \
               capable of forecasting water quality trends based on historical and current measurements.")
with col2:
    st.image(IMAGE1, use_column_width=True)

# Subheader and paragraph for additional information
st.divider()
st.subheader("Data Collection Process")
col3, col4 = st.columns(2)

with col3:
    # Data retrieval and training process
    st.write("Throughout our expeditions, we utilized cutting-edge sensors and automated sampling systems to gather \
               a comprehensive dataset on key water quality parameters, including temperature, pH, dissolved oxygen, and nutrient concentrations. \
               Each measurement was paired with precise GPS coordinates, ensuring that the data collected was both geospatially and \
               temporally accurate. This granular dataset laid the foundation for our machine learning model, enabling us to analyze \
               water quality patterns with fine detail and precision.")
    # Explanation of data preparation for model training
    st.write("To prepare the data for machine learning training, we performed a thorough cleaning and preprocessing step, removing \
               any inconsistencies and outliers to ensure the integrity of the dataset. The processed data was then divided into training \
               and testing sets, with the training set used to fit our machine learning model. We employed techniques like feature scaling, \
               normalization, and cross-validation to optimize the model's performance and reduce potential biases.")
    # Integration of real-time data and visual assets
    st.write("Real-time data from NOAA’s API was also integrated into the dataset, providing up-to-date environmental variables \
               for continuous model refinement. Alongside the raw data, we documented geographical features, marine life, and human activities \
               in the study areas through photographs and videos. These visuals not only enriched the understanding of water quality trends \
               but also served as powerful tools for community engagement and awareness regarding conservation efforts.")
with col4:
    st.image(IMAGE2, use_column_width=True)

st.divider()
col5, col6, col7 = st.columns([1, 1, 1])

with col5:
    st.image(IMAGE_MLH, width=200)
with col7:
    st.image(IMAGE_SH, width=300)

