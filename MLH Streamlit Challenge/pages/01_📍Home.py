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

IMAGE_BANNER = "media/ShellHacksBanner.jpg"
IMAGE_FIU_BANNER = "media/FIU_Banner.png"
IMAGE_MLH = "media/MLH.png"

st.set_page_config(page_title="ShellHacks: MLH Streamlit Challenge", layout="wide",
                   page_icon="🌊", initial_sidebar_state="expanded")

with st.sidebar:
    st.image(IMAGE_MLH, use_column_width=True)
    st.title("ShellHacks: MLH Streamlit Challenge")



st.image(IMAGE_BANNER, use_column_width=True)
st.title("Water Quality Predictor with Machine Learning and Data Visualization")
st.title("Home")
# Project overview paragraph
st.write("As soon-to-graduate computer science students participating in ShellHacks 2024,\
 we were driven by a shared commitment to addressing one of the world’s most critical challenges—access\
  to clean and safe water. The motivation behind developing a Streamlit website that leverages machine learning\
   for water quality forecasting stems from this urgency. Clean water is fundamental not only to public health,\
    but also to environmental preservation and sustainable economic development. However, many regions, both in \
    developed and developing countries, struggle with monitoring and managing water resources effectively.")

st.write("Our project was designed to bridge this gap by creating a tool that empowers communities, local governments,\
 and environmental organizations with a real-time, user-friendly platform for forecasting water quality. With machine\
  learning at its core, the website processes historical and real-time data to make accurate predictions about water safety,\
   offering insights that can help prevent waterborne diseases, reduce pollution, and support long-term environmental sustainability efforts.")

st.write("By building this solution, we hope to contribute to raising awareness about global water challenges while showcasing the potential\
 of modern technology in addressing them. Additionally, our project serves as a testament to the power of cross-disciplinary collaboration,\
  where computer science, data analytics, and environmental science come together to make a tangible impact.")

st.divider()
st.subheader("Impact on Society")
st.write("We want to leverage technology to address real-world problems, and water quality monitoring is an essential\
    aspect of environmental and public health. By building a system that can predict potential water contamination or\
     detect changes in quality, this team aims to provide an accessible tool for communities, researchers, and\
      policymakers to take proactive action in ensuring clean water supply. ")

# Data collection and analysis paragraph
st.divider()
st.subheader("Machine Learning and Innovation")
st.write("This hackathon offers the perfect opportunity to experiment with machine learning models in a meaningful\
    context. By analyzing water quality data (such as pH levels, dissolved oxygen, contaminants, etc.), we can create\
     a model that predicts future trends based on historical patterns and real-time inputs. This also allows us to apply\
      our knowledge of data science and machine learning to a practical and socially relevant problem.")

st.divider()
st.subheader("Accessibility and Usability")
st.write("Streamlit is a powerful yet simple framework that allows us to quickly create interactive web applications.\
    By using Streamlit, we can present complex data and predictions in a user-friendly interface, making the water quality\
     forecast tool accessible to a wide range of users, from government agencies to local communities or even individuals \
     concerned about their water supply.")

# Footer
st.divider()
col5, col6, col7 = st.columns([1, 1, 1])
with col6:
    st.image(IMAGE_FIU_BANNER, width=200)
