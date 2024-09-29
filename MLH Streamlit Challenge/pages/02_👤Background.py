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
