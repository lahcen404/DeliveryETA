import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


# ============================================================
# LOAD MODEL, SCALER AND DATA
# ============================================================

model = joblib.load("models/delivery_eta_pipeline.joblib")
scaler = joblib.load("models/scaler.joblib")

data = pd.read_csv("data/selected_features_data.csv")

y = data["Time_taken(min)"]


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DeliveryETA",
    page_icon="🚚",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🚚 DeliveryETA")
st.subheader("Prediction of Delivery Time")

st.write(
    "Enter the delivery information below to estimate the total "
    "delivery time in minutes."
)


# ============================================================
# INPUTS
# ============================================================

st.header("📋 Delivery Information")

col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "Delivery Person Age",
        min_value=20,
        max_value=39,
        value=30
    )

    rating = st.number_input(
        "Delivery Person Rating",
        min_value=1.0,
        max_value=5.0,
        value=4.7,
        step=0.1
    )

    distance = st.number_input(
        "Distance (km)",
        min_value=1.46,
        max_value=20.94,
        value=9.0,
        step=0.1
    )

    multiple_deliveries = st.selectbox(
        "Multiple Deliveries",
        [0, 1, 2, 3],
        index=1
    )


with col2:

    weather = st.selectbox(
        "Weather",
        [
            "Cloudy",
            "Fog",
            "Sandstorms",
            "Stormy",
            "Sunny",
            "Windy"
        ]
    )

    traffic = st.selectbox(
        "Road Traffic Density",
        [
            "High",
            "Jam",
            "Low",
            "Medium"
        ]
    )

    vehicle = st.selectbox(
        "Type of Vehicle",
        [
            "bicycle",
            "electric_scooter",
            "motorcycle",
            "scooter"
        ]
    )

    day = st.selectbox(
        "Day of Week",
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]
    )


with col3:

    order_hour = st.number_input(
        "Order Hour",
        min_value=0,
        max_value=23,
        value=17
    )

    pickup_hour = st.number_input(
        "Pickup Hour",
        min_value=0,
        max_value=23,
        value=17
    )

    pickup_delay = st.number_input(
        "Pickup Delay (minutes)",
        min_value=5,
        max_value=15,
        value=10
    )

    month = st.selectbox(
        "Month",
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December"
        ],
        index=1
    )


# ============================================================
# PREPARE INPUT
# ============================================================

def prepare_input():

    # --------------------------------------------------------
    # Numerical features
    # --------------------------------------------------------

    numerical_data = pd.DataFrame([{
        "Delivery_person_Age": age,
        "Delivery_person_Ratings": rating,
        "multiple_deliveries": multiple_deliveries,
        "distance": distance,
        "order_hour": order_hour,
        "pickup_hour": pickup_hour,
        "pickup_delay": pickup_delay,
        "month": month
    }])

    # Convert month name to number
    month_number = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12
    }

    numerical_data["month"] = numerical_data["month"].map(month_number)

    # --------------------------------------------------------
    # Scale numerical features
    # --------------------------------------------------------

    numerical_scaled = scaler.transform(numerical_data)

    numerical_scaled = pd.DataFrame(
        numerical_scaled,
        columns=[
            "Delivery_person_Age",
            "Delivery_person_Ratings",
            "multiple_deliveries",
            "distance",
            "order_hour",
            "pickup_hour",
            "pickup_delay",
            "month"
        ]
    )


    # --------------------------------------------------------
    # Categorical features
    # --------------------------------------------------------

    categorical_columns = [
        "Weatherconditions_Cloudy",
        "Weatherconditions_Fog",
        "Weatherconditions_Sandstorms",
        "Weatherconditions_Stormy",
        "Weatherconditions_Sunny",
        "Weatherconditions_Windy",

        "Road_traffic_density_High",
        "Road_traffic_density_Jam",
        "Road_traffic_density_Low",
        "Road_traffic_density_Medium",

        "Type_of_vehicle_bicycle",
        "Type_of_vehicle_electric_scooter",
        "Type_of_vehicle_motorcycle",
        "Type_of_vehicle_scooter",

        "day_of_week_Friday",
        "day_of_week_Monday",
        "day_of_week_Saturday",
        "day_of_week_Sunday",
        "day_of_week_Thursday",
        "day_of_week_Tuesday",
        "day_of_week_Wednesday"
    ]

    categorical_data = pd.DataFrame(
        np.zeros(
            (1, len(categorical_columns))
        ),
        columns=categorical_columns
    )


    # --------------------------------------------------------
    # Set selected categorical values to 1
    # --------------------------------------------------------

    categorical_data[
        f"Weatherconditions_{weather}"
    ] = 1

    categorical_data[
        f"Road_traffic_density_{traffic}"
    ] = 1

    categorical_data[
        f"Type_of_vehicle_{vehicle}"
    ] = 1

    categorical_data[
        f"day_of_week_{day}"
    ] = 1


    # --------------------------------------------------------
    # Combine numerical + categorical features
    # --------------------------------------------------------

    final_input = pd.concat(
        [
            numerical_scaled,
            categorical_data
        ],
        axis=1
    )


    # --------------------------------------------------------
    # Exact training feature order
    # --------------------------------------------------------

    training_columns = [
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "multiple_deliveries",
        "distance",
        "order_hour",
        "pickup_hour",
        "pickup_delay",
        "month",

        "Weatherconditions_Cloudy",
        "Weatherconditions_Fog",
        "Weatherconditions_Sandstorms",
        "Weatherconditions_Stormy",
        "Weatherconditions_Sunny",
        "Weatherconditions_Windy",

        "Road_traffic_density_High",
        "Road_traffic_density_Jam",
        "Road_traffic_density_Low",
        "Road_traffic_density_Medium",

        "Type_of_vehicle_bicycle",
        "Type_of_vehicle_electric_scooter",
        "Type_of_vehicle_motorcycle",
        "Type_of_vehicle_scooter",

        "day_of_week_Friday",
        "day_of_week_Monday",
        "day_of_week_Saturday",
        "day_of_week_Sunday",
        "day_of_week_Thursday",
        "day_of_week_Tuesday",
        "day_of_week_Wednesday"
    ]

    final_input = final_input[training_columns]

    return final_input


# ============================================================
# PREDICTION
# ============================================================

st.header("🔮 Delivery Time Prediction")

if st.button(
    "Predict Delivery Time",
    type="primary"
):

    input_data = prepare_input()

    prediction = model.predict(input_data)[0]

    prediction = max(0, prediction)

    st.metric(
        label="Estimated Delivery Time",
        value=f"{prediction:.1f} minutes"
    )

    st.success(
        f"Estimated delivery time: **{round(prediction)} minutes**"
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.header("📈 Model Performance")

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "MAE",
        "3.57 min"
    )

    st.caption(
        "Average prediction error"
    )


with col2:

    st.metric(
        "RMSE",
        "4.55 min"
    )

    st.caption(
        "Penalizes larger errors"
    )


with col3:

    st.metric(
        "R²",
        "76.73%"
    )

    st.caption(
        "Variance explained by the model"
    )


st.info(
    "On the test dataset, the model's predictions are on average "
    "about 3.57 minutes away from the actual delivery time."
)


# ============================================================
# VISUALIZATION
# ============================================================

st.header("📊 Data Visualization")

st.subheader("Delivery Time Distribution")

fig, ax = plt.subplots(figsize=(8, 4))

ax.hist(
    y,
    bins=20,
    edgecolor="black"
)

ax.set_xlabel("Delivery Time (minutes)")
ax.set_ylabel("Number of Deliveries")
ax.set_title("Distribution of Delivery Times")

st.pyplot(fig)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🚚 DeliveryETA")

    st.write(
        "Machine Learning model for predicting delivery time."
    )

    st.divider()

    st.subheader("Model")

    st.write(
        "Random Forest Regressor"
    )

    st.write(
        "Optimized with RandomizedSearchCV"
    )

    st.divider()

    st.subheader("Performance")

    st.write("MAE: **3.57 min**")
    st.write("RMSE: **4.55 min**")
    st.write("R²: **76.73%**")

    st.divider()

    st.caption(
        "DeliveryETA — Machine Learning Project"
    )