import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
from sklearn.utils import resample
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Bus Travel Time Predictor",
    page_icon="🚌",
    layout="wide"
)


# Load the saved model and data
@st.cache_resource
def load_model_data():
    with open('gradient_boosting_travel_time_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('target_encoder_transformer.pkl', 'rb') as f:
        transformer = pickle.load(f)
    with open('feature_lists.pkl', 'rb') as f:
        feature_lists = pickle.load(f)
    with open('bus_travel_data.pkl', 'rb') as f:
        df = pickle.load(f)
    return model, transformer, feature_lists, df


model, transformer, feature_lists, df = load_model_data()
categorical_features = feature_lists['categorical_features']
numerical_features = feature_lists['numerical_features']

# Title and description
st.title("🚌 Bus Travel Time Predictor")
st.markdown("""
This application predicts bus travel times based on various factors like route, time, weather, and passenger load conditions.
The prediction model is built using Gradient Boosting with Target Encoding.
""")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.header("Route & Time Information")

    # Route selection
    route_id = st.selectbox(
        "Select Route",
        sorted(df['route_id'].unique())
    )

    # Date selection
    date = st.date_input(
        "Date",
        datetime.date.today()
    )

    # Time selection
    time = st.time_input(
        "Scheduled Start Time",
        datetime.time(8, 0)
    )

    # Day of week (calculated from date)
    day_of_week = date.strftime("%A")
    st.write(f"Day of Week: {day_of_week}")

    # Is holiday
    is_holiday = st.checkbox("Is Holiday")

with col2:
    st.header("Conditions")

    # Weather conditions
    weather_condition = st.selectbox(
        "Weather Condition",
        sorted(df['weather_condition'].unique())
    )

    # Passenger load slider
    passenger_load = st.slider(
        "Passenger Load (0.0 to 1.0)",
        0.0, 1.0, 0.5, 0.05
    )

    # Additional route info (readonly)
    route_info = {
        "R01": {"route_length_km": 18.5, "num_stops": 12},
        "R02": {"route_length_km": 25.0, "num_stops": 18},
        "R03": {"route_length_km": 12.0, "num_stops": 8},
        "R04": {"route_length_km": 30.0, "num_stops": 22},
        "R05": {"route_length_km": 20.0, "num_stops": 15}
    }

    st.write(f"Route Length: {route_info[route_id]['route_length_km']} km")
    st.write(f"Number of Stops: {route_info[route_id]['num_stops']}")


# Function to find similar trips in the dataset
def find_similar_trips(df, route_id, hour, day_of_week, weather_condition):
    similar_trips = df[
        (df['route_id'] == route_id) &
        (df['hour'] == hour) &
        (df['day_of_week'] == day_of_week) &
        (df['weather_condition'] == weather_condition)
        ]

    if len(similar_trips) == 0:
        # If no exact matches, relax the weather condition constraint
        similar_trips = df[
            (df['route_id'] == route_id) &
            (df['hour'] == hour) &
            (df['day_of_week'] == day_of_week)
            ]

    if len(similar_trips) == 0:
        # If still no matches, relax the day of week constraint
        similar_trips = df[
            (df['route_id'] == route_id) &
            (df['hour'] == hour)
            ]

    return similar_trips


# Predict button
if st.button("🔍 Predict Travel Time"):
    # Format the date and time
    formatted_date = date.strftime("%d-%m-%Y")
    formatted_time = time.strftime("%H:%M:%S")

    # Determine time period
    hour = time.hour
    minute = time.minute

    if 6 <= hour <= 9:
        time_period = "morning_peak"
    elif 10 <= hour <= 15:
        time_period = "midday"
    elif 16 <= hour <= 18:
        time_period = "evening_peak"
    else:
        time_period = "off_peak"

    # Automatically determine traffic level based on time period
    if time_period == "morning_peak" or time_period == "evening_peak":
        if day_of_week in ["Saturday", "Sunday"]:
            traffic_level = 1.0  # Normal traffic on weekends even during peak hours
        else:
            traffic_level = 1.5  # Heavy traffic on weekdays during peak hours
    else:
        traffic_level = 1.0  # Normal traffic during off-peak hours

    # Create input data
    input_data = pd.DataFrame({
        'route_id': [route_id],
        'month': [date.month],
        'day_of_week': [day_of_week],
        'hour': [hour],
        'minute': [minute],
        'is_holiday': [1 if is_holiday else 0],
        'is_weekend': [1 if day_of_week in ["Saturday", "Sunday"] else 0],
        'route_length_km': [route_info[route_id]["route_length_km"]],
        'num_stops': [route_info[route_id]["num_stops"]],
        'weather_condition': [weather_condition],
        'traffic_level': [traffic_level],  # Automatically determined
        'passenger_load': [passenger_load],
        'time_period': [time_period]
    })

    # Get prediction
    prediction = model.predict(input_data)[0]

    # Find similar trips in the dataset
    similar_trips = find_similar_trips(df, route_id, hour, day_of_week, weather_condition)
    actual_times = similar_trips['travel_time_minutes'].values if len(similar_trips) > 0 else []

    # Calculate simple confidence interval based on similar trips
    if len(actual_times) > 0:
        std_dev = np.std(actual_times)
        lower_bound = prediction - 1.96 * std_dev  # 95% confidence interval
        upper_bound = prediction + 1.96 * std_dev
    else:
        # Use a default uncertainty of 15% if no similar trips
        lower_bound = prediction * 0.85
        upper_bound = prediction * 1.15

    # Display prediction with confidence interval
    st.header("Prediction Results")

    # Create columns for predicted and actual
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Predicted Travel Time")
        st.markdown(f"""
            <div style="background-color:#f0f8ff; padding:20px; border-radius:10px;">
                <h1 style="text-align:center; color:#1e90ff;">{prediction:.2f} minutes</h1>
                <p style="text-align:center;">(Range: {lower_bound:.2f} - {upper_bound:.2f} minutes)</p>
            </div>
            """, unsafe_allow_html=True)

        # Display the automatically determined traffic level
        if traffic_level > 1.0:
            st.write(f"Automatically detected heavy traffic (level: {traffic_level})")
        else:
            st.write(f"Normal traffic conditions detected (level: {traffic_level})")

        # Travel speed
        speed = route_info[route_id]["route_length_km"] / (prediction / 60)
        st.write(f"Average Speed: {speed:.2f} km/h")

        # Expected arrival
        arrival_time = (datetime.datetime.combine(date, time) +
                        datetime.timedelta(minutes=prediction)).time()
        st.write(f"Expected Arrival Time: {arrival_time.strftime('%H:%M:%S')}")

    with col2:
        if len(actual_times) > 0:
            st.subheader("Actual Travel Times (Similar Trips)")
            mean_actual = np.mean(actual_times)
            st.markdown(f"""
            <div style="background-color:#f0fff0; padding:20px; border-radius:10px;">
                <h1 style="text-align:center; color:#228b22;">{mean_actual:.2f} minutes</h1>
                <p style="text-align:center;">(Average of {len(actual_times)} similar trips)</p>
            </div>
            """, unsafe_allow_html=True)

            # Show range
            min_actual = np.min(actual_times)
            max_actual = np.max(actual_times)
            st.write(f"Range: {min_actual:.2f} - {max_actual:.2f} minutes")
        else:
            st.subheader("No Similar Trips Found")
            st.markdown("""
            <div style="background-color:#fff0f0; padding:20px; border-radius:10px; text-align:center;">
                <p>No historical data available for similar trips</p>
            </div>
            """, unsafe_allow_html=True)

    # Visualization
    st.subheader("Visualization")

    # Create a figure with comparison if we have actual times
    fig, ax = plt.subplots(figsize=(10, 6))

    if len(actual_times) > 0:
        # Plot histogram of actual times if available
        sns.histplot(actual_times, kde=True, ax=ax, color='green', alpha=0.6, label='Historical Trips')
        ax.axvline(x=mean_actual, color='darkgreen', linestyle='--', linewidth=2, label='Historical Average')

    # Plot the prediction
    ax.axvline(x=prediction, color='blue', linestyle='-', linewidth=2, label='Prediction')

    ax.set_xlabel('Travel Time (minutes)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Predicted vs. Historical Travel Times for {route_id} on {day_of_week} at {hour}:00')
    ax.legend()

    st.pyplot(fig)

    # Additional insights
    st.subheader("Additional Insights")

    # Time period analysis
    st.write(f"Time Period: **{time_period}**")

    time_period_avg = df[df['time_period'] == time_period]['travel_time_minutes'].mean()
    st.write(f"Average travel time during this time period: {time_period_avg:.2f} minutes")

    # Weather impact
    weather_impact = df.groupby('weather_condition')['travel_time_minutes'].mean()
    st.write(
        f"Weather Impact: {weather_condition} conditions typically affect travel times by {(weather_impact[weather_condition] / time_period_avg - 1) * 100:.1f}% compared to overall average")

    # Suggestion
    if hour in [7, 8, 17, 18] and day_of_week not in ["Saturday", "Sunday"]:
        st.warning("⚠️ You're traveling during peak hours. Consider adjusting your travel time if possible.")

# Add a section showing model performance metrics
st.header("Model Performance")
st.write(
    "The prediction model is based on Gradient Boosting with Target Encoding and has the following performance metrics:")
st.write(f"- Mean Absolute Error (MAE): 6.28 minutes")
st.write(f"- Root Mean Squared Error (RMSE): 9.20 minutes")
st.write(f"- R-squared (R²): 0.897")

# Footer
st.markdown("---")
st.markdown("Developed using Gradient Boosting with Target Encoding | Data Last Updated: April 2025")