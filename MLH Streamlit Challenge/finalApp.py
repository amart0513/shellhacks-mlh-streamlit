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

st.set_page_config(page_title="ShellHacks: MLH Streamlit Challenge", layout="wide",
                   page_icon="🌊", initial_sidebar_state="expanded")

# NOAA API endpoint for buoy data
API_URL = "https://www.ndbc.noaa.gov/data/realtime2/<station_id>.txt"

# Images for project
IMAGE_JESUS = "media/JesusPic.jpg"
IMAGE_ANGIE = "media/AngiePic.jpg"
IMAGE_FIU = "media/FIU_LOGO.png"
IMAGE_BANNER = "media/ShellHacksBanner.jpg"
IMAGE1 = "media/BBC_Research1.jpg"
IMAGE2 = "media/BBC_Research2.jpg"
IMAGE_FIU_BANNER = "media/FIU_Banner.png"
IMAGE_MLH = "media/MLH.png"
IMAGE_SH = "media/SH.png"

MAJORS = [
    "",  # Placeholder for an empty selection
    "Accounting",
    "Aerospace Engineering",
    "Agricultural Science",
    "Anthropology",
    "Architecture",
    "Art History",
    "Biochemistry",
    "Biomedical Engineering",
    "Chemical Engineering",
    "Civil Engineering",
    "Computer Science",
    "Criminal Justice",
    "Cybersecurity",
    "Dentistry",
    "Economics",
    "Electrical Engineering",
    "Environmental Science",
    "Film Studies",
    "Finance",
    "Graphic Design",
    "History",
    "Industrial Engineering",
    "International Relations",
    "Journalism",
    "Linguistics",
    "Management",
    "Marketing",
    "Mathematics",
    "Mechanical Engineering",
    "Medicine",
    "Music",
    "Nursing",
    "Nutrition",
    "Pharmacy",
    "Philosophy",
    "Physics",
    "Political Science",
    "Psychology",
    "Public Health",
    "Sociology",
    "Software Engineering",
    "Statistics",
    "Theater",
    "Urban Planning",
    "Veterinary Science",
    "Web Development"
]


# create change font size button
# add images
# sidebar

# Define function to load media
def load_media(column, file_path, caption):
    with column:
        if file_path.endswith(".jpeg") or file_path.endswith(".PNG") or file_path.endswith(".jpg"):
            column.image(file_path, caption=caption, width=200)
        elif file_path.endswith(".mp4"):
            column.video(file_path)
        elif file_path.endswith(".mp3"):
            column.audio(file_path)


# To make scatter plots
def scatter_plots(df):
    st.subheader("Scatter Plot")
    fig = px.scatter(df, x="Depth m", y="Temp °C", size="pH", color="ODO mg/L")
    st.plotly_chart(fig)


# To make map plots
def maps(df):
    st.subheader("Maps")
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        fig = px.scatter_mapbox(df, lat="Latitude", lon="Longitude",
                                hover_data=["Depth m", "pH", "Temp °C", "ODO mg/L"],
                                zoom=15)
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig)
    else:
        st.error("Missing 'Latitude' or 'Longitude' columns in data.")


# To make line graphs
def line_plots(df):
    st.subheader("Line Plot")
    color = st.color_picker("Choose a color", "#081E3F")
    fig = px.line(df, x=df.index, y="ODO mg/L")
    fig.update_traces(line_color=color)
    st.plotly_chart(fig)


# To make 3D plots
def three_d_plots(df):
    st.subheader("3D Plot")
    fig = px.scatter_3d(df, x="Longitude", y="Latitude", z="Depth m", color="ODO mg/L")
    fig.update_scenes(zaxis_autorange="reversed")
    st.plotly_chart(fig)


# Plot the correlation heatmap
def plot_correlation_heatmap(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt)


# To make pair plots
def plot_pairplot(df, target):
    sns.pairplot(df, hue=target)
    st.pyplot(plt)


# Making predicted vs actual plots
def plot_predictions_vs_actual(y_test, predictions):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predictions vs Actual')
    st.pyplot(plt)


# Make plot residuals for training model
def plot_residuals(y_test, predictions):
    residuals = y_test - predictions
    plt.figure(figsize=(8, 6))
    plt.scatter(predictions, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    st.pyplot(plt)


# Find the error distribution from model
def plot_error_distribution(y_test, predictions):
    residuals = y_test - predictions
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True)
    plt.title('Distribution of Errors')
    st.pyplot(plt)


# Make a learning curve plot from the model
def plot_learning_curve(model, X, y):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5)
    train_mean = train_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, label='Training Score')
    plt.plot(train_sizes, test_mean, label='Validation Score')
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title('Learning Curves')
    plt.legend()
    st.pyplot(plt)


# Raw data from dataframe (csv)
def raw_data(df):
    st.subheader("Fetched Data")
    st.dataframe(df)
    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe())


# Define functions to render different pages
def render_home_page():
    st.image(IMAGE_BANNER, use_column_width=True)
    st.title("FlowCast: Real-Time Water Monitoring and Prediction")
    st.subheader("Home")
    # Project overview paragraph
    st.write("As a soon to graduate computer science students participating in ShellHacks 2024, our motive for creating a \
    Streamlit website that uses machine learning to forecast water quality stems from the desire to tackle a pressing global\
     issue—access to clean water. Water quality is crucial for public health, environmental sustainability, and economic development,\
      yet many communities around the world face challenges in monitoring and managing this resource effectively.")

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
    with col5:
        st.image(IMAGE_FIU_BANNER, width=200)


def render_about():
    st.title("About Us")
    st.subheader("Get to Know the Team!")
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    load_media(col2, IMAGE_JESUS, "Jesus Elespuru, Senior, Back-End Developer, Data-Collection, Florida International "
                                  "University")
    load_media(col3, IMAGE_ANGIE,
               "Angie Martinez, Senior, Front-End Developer, Data-Collection, Florida International University")
    st.divider()
    col5, col6, col7 = st.columns([1, 1, 1])

    with col5:
        st.image(IMAGE_FIU_BANNER, width=200)


def render_data():
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully.")
    else:
        df = pd.read_csv("mission156-complete.csv")
        st.info("Using default dataset.")

    # Data validation
    required_columns = ['Depth m', 'Temp °C', 'pH', 'ODO mg/L']
    for column in required_columns:
        if column not in df.columns:
            st.error(f"Missing column: {column}. Please upload a valid CSV file.")
            return

    # Handle NaN values
    if df.isnull().values.any():
        st.warning("Data contains NaN values. Please clean your data.")

    st.header("Water Quality Monitoring")
    st.subheader("Data Analysis")

    # Create sliders dynamically based on the data
    min_depth, max_depth = df["Depth m"].min(), df["Depth m"].max()
    min_temp, max_temp = df["Temp °C"].min(), df["Temp °C"].max()
    min_ph, max_ph = df["pH"].min(), df["pH"].max()

    selected_depth = st.slider("Select Depth (m)", min_value=min_depth, max_value=max_depth,
                               value=(min_depth, max_depth))
    selected_temp = st.slider("Select Temperature (°C)", min_value=min_temp, max_value=max_temp,
                              value=(min_temp, max_temp))
    selected_ph = st.slider("Select pH", min_value=min_ph, max_value=max_ph, value=(min_ph, max_ph))

    # Filter data based on user selection
    filtered_df = df[(df["Depth m"].between(selected_depth[0], selected_depth[1])) &
                     (df["Temp °C"].between(selected_temp[0], selected_temp[1])) &
                     (df["pH"].between(selected_ph[0], selected_ph[1]))]

    # Create tabs for different visualizations
    tabs = st.tabs(
        ["Scatter Plots", "Maps", "Line", "3D Plots", "Raw Data", "ML and Data Visualizations"])

    # Prepare features and target variable
    features = df[['Depth m', 'Temp °C', 'pH', 'ODO mg/L']]
    target = df['ODO mg/L']

    model_file = 'data.pkl'
    mse, r2 = None, None  # Initialize metrics

    # Model training and metric calculation
    if not os.path.exists(model_file):
        # Plot correlation heatmap
        plot_correlation_heatmap(df)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.6, random_state=42)

        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Test the model
        predictions = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # Save model
        dump(model, model_file)
        st.success("Model trained and saved.")

    else:
        model = load(model_file)

        # Use filtered data for predictions
        predictions = model.predict(filtered_df[['Depth m', 'Temp °C', 'pH', 'ODO mg/L']])

        # Calculate metrics on filtered data
        mse = mean_squared_error(filtered_df['ODO mg/L'], predictions)
        r2 = r2_score(filtered_df['ODO mg/L'], predictions)

    # Visualization Tabs
    with tabs[0]:  # Scatter Plots
        scatter_plots(filtered_df)
    with tabs[1]:  # Maps
        maps(filtered_df)
    with tabs[2]:  # Line
        line_plots(filtered_df)
    with tabs[3]:  # 3D Plots
        three_d_plots(filtered_df)
    with tabs[4]:  # Raw Data
        raw_data(filtered_df)

    # Machine Learning Visualizations
    with tabs[5]:  # ML and Data Visualizations
        if mse is not None and r2 is not None:
            # Display metrics only if they are calculated
            st.subheader("Machine Learning Model Metrics")
            st.write(f"Mean Squared Error (MSE): {mse:.2f}")
            st.write(f"R² Score: {r2:.2f}")

            # Plot predicted vs actual graphs
            st.subheader("Predicted vs Actual")
            plot_predictions_vs_actual(filtered_df['ODO mg/L'], predictions)

            # Plot the error distribution after predicting
            st.subheader("Error Distribution")
            plot_error_distribution(filtered_df['ODO mg/L'], predictions)

            # Plot residuals
            st.subheader("Residuals")
            plot_residuals(filtered_df['ODO mg/L'], predictions)
        else:
            st.warning("Metrics not available. Please train the model first.")


def render_background():
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
        st.image(IMAGE_FIU_BANNER, width=200)


def render_sign_up():
    st.title('Sign Up to Learn More')
    st.write('Please enter your information below:')

    with st.form("Registration", clear_on_submit=True):
        name = st.text_input("Name:")
        email = st.text_input("Email:")
        major = st.selectbox("Major:",
                             options=MAJORS)
        level = st.selectbox("Degree Level:", options=["", "Undergrad", "Masters", "PhD", "Other"])
        subscribe = st.checkbox("Do you want to know about future events?")
        submit = st.form_submit_button("Submit")
        if (name and email and submit and subscribe and level) or (name and email and submit and level):
            st.success(f"{name}, {level} in {major}, is now registered")
        elif submit:
            st.warning(f"{name}, {level} in {major}, is NOT registered")
        else:
            st.info("Please Fill out the form")
    st.divider()
    col5, col6, col7 = st.columns([1, 1, 1])

    with col5:
        st.image(IMAGE_FIU_BANNER, width=200)


# Fetching API from NOAA
def render_API():
    st.title("Real-Time Data from National Oceanic and Atmospheric Administration (NOAA)")

    # Define station IDs and their titles
    stations = {
        '41122': 'Hollywood Beach, FL',
        '41114': 'Fort Pierce, FL',
        '41010': 'Cape Canaveral, FL',
        '42036': 'Tampa, FL',
        '41070': 'Daytona Beach, FL'
    }

    # Sidebar for selecting a single station ID
    selected_station = st.sidebar.selectbox(
        "Select Station ID",
        list(stations.keys()),
        index=0  # Default selection (first station)
    )

    if st.button('Fetch Real-Time Data'):
        response = requests.get(API_URL.replace('<station_id>', selected_station))
        if response.status_code == 200:
            data = response.text.splitlines()
            columns = ['YY', 'MM', 'DD', 'hh', 'mm', 'WDIR', 'WSPD', 'GST', 'WVHT', 'DPD', 'APD', 'MWD',
                       'PRES', 'ATMP', 'WTMP', 'DEWP', 'VIS', 'PTDY', 'TIDE']

            # Skip header rows if necessary and construct the DataFrame
            df_api = pd.DataFrame([x.split() for x in data[2:] if x.strip() != ''], columns=columns)

            # Convert WTMP to numeric, forcing errors to NaN
            df_api['WTMP'] = pd.to_numeric(df_api['WTMP'], errors='coerce')

            # Display the title for the current station
            st.subheader(stations[selected_station])

            # Display the raw data or any visualizations you want
            raw_data(df_api)

            # Check if we have valid data for water temperature
            valid_data = df_api['WTMP'].dropna()
            if not valid_data.empty:
                # Create a line chart using matplotlib
                plt.figure(figsize=(10, 5))
                plt.plot(valid_data.index, valid_data, marker='o', linestyle='-', color='b')
                plt.title(f'Water Temperature at {stations[selected_station]}')
                plt.xlabel('Index')
                plt.ylabel('Water Temperature (°C)')
                plt.xticks(rotation=45)
                plt.grid()
                plt.legend(['Water Temperature'])

                # Display the matplotlib chart in Streamlit
                st.pyplot(plt)
            else:
                st.warning(f"No valid water temperature data available for the selected station.")
        else:
            st.error(f"Failed to retrieve data for station ID {selected_station}. Please try again.")


# Load and predict the water quality prediction
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

        if 'ODO mg/L' in df.columns:
            true_values = df['ODO mg/L']
            mse = mean_squared_error(true_values, predictions)
            st.write(f"**Mean Squared Error (MSE):** {mse}")

            # Scatter Plot
            st.subheader("Actual vs Predicted Dissolved Oxygen (ODO mg/L)")
            plt.figure(figsize=(10, 6))
            plt.scatter(true_values, predictions, alpha=0.6)
            plt.plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], 'r--', lw=2)
            plt.xlabel("Actual ODO mg/L")
            plt.ylabel("Predicted ODO mg/L")
            plt.title("Scatter Plot of Actual vs Predicted ODO")
            st.pyplot(plt)
            plt.clf()  # Clear the figure

            # Error Distribution
            st.subheader("Error Distribution")
            errors = true_values - predictions
            plt.figure(figsize=(10, 6))
            sns.histplot(errors, bins=30, kde=True)
            plt.xlabel("Prediction Error (Actual - Predicted ODO mg/L)")
            plt.title("Error Distribution of Predictions")
            st.pyplot(plt)
            plt.clf()  # Clear the figure

            # Line Plot
            st.subheader("Line Plot of Actual and Predicted ODO over Index")
            plt.figure(figsize=(10, 6))
            plt.plot(prediction_df.index, true_values, label='Actual ODO mg/L', color='blue')
            plt.plot(prediction_df.index, predictions, label='Predicted ODO mg/L', color='orange', linestyle='--')
            plt.xlabel("Index")
            plt.ylabel("ODO mg/L")
            plt.title("Line Plot of Actual vs Predicted ODO")
            plt.legend()
            st.pyplot(plt)

    else:
        st.error(f"Model file {model_file} not found. Train the model first.")



# Load data and render prediction page
def render_predictions_water_quality():
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


with st.sidebar:
    st.image(IMAGE_MLH, width=100)
    st.title("ShellHacks: MLH Streamlit Challenge")
    page = st.sidebar.selectbox("Navigate", options=["Home", "Background", "About Us", "Data Analysis",
                                                     "NOAA API Retrieval", "Predictive Analysis with ML", "Sign Up"])

# Conditionally render pages based on selection
if page == "Home":
    render_home_page()
elif page == "About Us":
    render_about()
elif page == "Data Analysis":
    render_data()
elif page == "Background":
    render_background()
elif page == "Sign Up":
    render_sign_up()
elif page == "NOAA API Retrieval":
    render_API()
elif page == "Predictive Analysis with ML":
    render_predictions_water_quality()
