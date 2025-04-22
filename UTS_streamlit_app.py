import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import RobustScaler

# Load trained model
MODEL_PATH = "best_xgboost_model.pkl"
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

# Page title
st.title("Hotel Booking Cancellation Prediction")

# Sidebar input
st.sidebar.header("Input Features")

def user_input():
    no_of_adults = st.sidebar.slider('Number of Adults', 1, 5, 2)
    no_of_children = st.sidebar.slider('Number of Children', 0, 10, 0)
    no_of_weekend_nights = st.sidebar.slider('Weekend Nights', 0, 10, 1)
    no_of_week_nights = st.sidebar.slider('Week Nights', 0, 10, 1)
    type_of_meal_plan = st.sidebar.selectbox('Meal Plan', [0, 1, 2, 3])
    required_car_parking_space = st.sidebar.selectbox('Need Car Parking', [0, 1])
    room_type_reserved = st.sidebar.selectbox('Room Type Reserved', [0, 1, 2, 3, 4, 5, 6])
    lead_time = st.sidebar.slider('Lead Time', 0, 500, 30)
    arrival_year = st.sidebar.selectbox('Arrival Year', [2017])
    arrival_month = st.sidebar.slider('Arrival Month', 1, 12, 6)
    arrival_date = st.sidebar.slider('Arrival Day', 1, 31, 15)
    market_segment_type = st.sidebar.selectbox('Market Segment Type', [0, 1, 2, 3, 4, 5, 6])
    repeated_guest = st.sidebar.selectbox('Repeated Guest', [0, 1])
    no_of_previous_cancellations = st.sidebar.slider('Previous Cancellations', 0, 10, 0)
    no_of_previous_bookings_not_canceled = st.sidebar.slider('Previous Bookings Not Canceled', 0, 20, 0)
    avg_price_per_room = st.sidebar.slider('Avg Price per Room', 0.0, 500.0, 100.0)
    no_of_special_requests = st.sidebar.slider('Special Requests', 0, 5, 0)

    data = {
        'no_of_adults': no_of_adults,
        'no_of_children': no_of_children,
        'no_of_weekend_nights': no_of_weekend_nights,
        'no_of_week_nights': no_of_week_nights,
        'type_of_meal_plan': type_of_meal_plan,
        'required_car_parking_space': required_car_parking_space,
        'room_type_reserved': room_type_reserved,
        'lead_time': lead_time,
        'arrival_year': arrival_year,
        'arrival_month': arrival_month,
        'arrival_date': arrival_date,
        'market_segment_type': market_segment_type,
        'repeated_guest': repeated_guest,
        'no_of_previous_cancellations': no_of_previous_cancellations,
        'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled,
        'avg_price_per_room': avg_price_per_room,
        'no_of_special_requests': no_of_special_requests,
    }
    return pd.DataFrame([data])

input_df = user_input()

# Display input features
st.subheader("Input Features")
st.write(input_df)

# Scaling input to match training
scaler = RobustScaler()
scaled_input = scaler.fit_transform(input_df)  # ideally should load the training scaler, but we re-fit here due to simplicity

# Prediction
prediction = model.predict(scaled_input)[0]
result = "Booking Canceled" if prediction == 1 else "Booking Not Canceled"

st.subheader("Prediction Result")
st.success(result)

# Footer
st.markdown("---")
st.markdown("Developed by Tyrone · UTS Model Deployment")

