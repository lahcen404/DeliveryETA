import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# 1. LOAD MODEL + SCALER
# ============================================================

model = joblib.load("models/delivery_eta_pipeline.joblib")
scaler = joblib.load("models/scaler.joblib")


# ============================================================
# 2. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DeliveryETA",
    page_icon="🚚",
    layout="centered"
)


# ============================================================
# 3. TITLE
# ============================================================

st.title("🚚 DeliveryETA")

st.write(
    "Predict the estimated delivery time in minutes "
    "using delivery, traffic, weather and order information."
)

st.divider()


# ============================================================
# 4. USER INPUTS
# ============================================================

st.subheader("📦 Delivery Information")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# LEFT COLUMN
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# RIGHT COLUMN
# ------------------------------------------------------------

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

    day_of_week = st.selectbox(
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


# ============================================================
# 5. TIME INFORMATION
# ============================================================

st.subheader("🕐 Time Information")

col1, col2, col3 = st.columns(3)


with col1:

    order_hour = st.number_input(
        "Order Hour",
        min_value=0,
        max_value=23,
        value=17
    )


with col2:

    pickup_hour = st.number_input(
        "Pickup Hour",
        min_value=0,
        max_value=23,
        value=17
    )


with col3:

    pickup_delay = st.number_input(
        "Pickup Delay (min)",
        min_value=5,
        max_value=15,
        value=10
    )


month = st.selectbox(
    "Month",
    list(range(1, 13)),
    index=1
)


st.divider()


# ============================================================
# 6. CREATE PREPROCESSED INPUT
# ============================================================

def prepare_input():

    """
    Convert the Streamlit inputs into the same
    29 features used during model training.
    """

    # --------------------------------------------------------
    # Numerical features
    # --------------------------------------------------------

    numerical_data = np.array([[
        age,
        rating,
        multiple_deliveries,
        distance,
        order_hour,
        pickup_hour,
        pickup_delay,
        month
    ]])

    # Apply the scaler used during training
    numerical_scaled = scaler.transform(numerical_data)

    numerical_columns = [
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "multiple_deliveries",
        "distance",
        "order_hour",
        "pickup_hour",
        "pickup_delay",
        "month"
    ]

    numerical_df = pd.DataFrame(
        numerical_scaled,
        columns=numerical_columns
    )


    # --------------------------------------------------------
    # Categorical features
    # --------------------------------------------------------

    categorical_columns = [

        # Weather
        "Weatherconditions_Cloudy",
        "Weatherconditions_Fog",
        "Weatherconditions_Sandstorms",
        "Weatherconditions_Stormy",
        "Weatherconditions_Sunny",
        "Weatherconditions_Windy",

        # Traffic
        "Road_traffic_density_High",
        "Road_traffic_density_Jam",
        "Road_traffic_density_Low",
        "Road_traffic_density_Medium",

        # Vehicle
        "Type_of_vehicle_bicycle",
        "Type_of_vehicle_electric_scooter",
        "Type_of_vehicle_motorcycle",
        "Type_of_vehicle_scooter",

        # Day
        "day_of_week_Friday",
        "day_of_week_Monday",
        "day_of_week_Saturday",
        "day_of_week_Sunday",
        "day_of_week_Thursday",
        "day_of_week_Tuesday",
        "day_of_week_Wednesday"
    ]

    categorical_df = pd.DataFrame(
        0,
        index=[0],
        columns=categorical_columns
    )


    # --------------------------------------------------------
    # Set selected categories to 1
    # --------------------------------------------------------

    categorical_df[
        f"Weatherconditions_{weather}"
    ] = 1

    categorical_df[
        f"Road_traffic_density_{traffic}"
    ] = 1

    categorical_df[
        f"Type_of_vehicle_{vehicle}"
    ] = 1

    categorical_df[
        f"day_of_week_{day_of_week}"
    ] = 1


    # --------------------------------------------------------
    # Combine numerical + categorical features
    # --------------------------------------------------------

    final_input = pd.concat(
        [
            numerical_df,
            categorical_df
        ],
        axis=1
    )


    # --------------------------------------------------------
    # Exact training column order
    # --------------------------------------------------------

    expected_columns = [

        # Numerical
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "multiple_deliveries",
        "distance",
        "order_hour",
        "pickup_hour",
        "pickup_delay",
        "month",

        # Weather
        "Weatherconditions_Cloudy",
        "Weatherconditions_Fog",
        "Weatherconditions_Sandstorms",
        "Weatherconditions_Stormy",
        "Weatherconditions_Sunny",
        "Weatherconditions_Windy",

        # Traffic
        "Road_traffic_density_High",
        "Road_traffic_density_Jam",
        "Road_traffic_density_Low",
        "Road_traffic_density_Medium",

        # Vehicle
        "Type_of_vehicle_bicycle",
        "Type_of_vehicle_electric_scooter",
        "Type_of_vehicle_motorcycle",
        "Type_of_vehicle_scooter",

        # Day
        "day_of_week_Friday",
        "day_of_week_Monday",
        "day_of_week_Saturday",
        "day_of_week_Sunday",
        "day_of_week_Thursday",
        "day_of_week_Tuesday",
        "day_of_week_Wednesday"
    ]

    final_input = final_input[expected_columns]

    return final_input


# ============================================================
# 7. PREDICTION
# ============================================================

if st.button(
    "🚀 Predict Delivery Time",
    use_container_width=True
):

    try:

        # Prepare input
        input_data = prepare_input()

        # Make prediction
        prediction = model.predict(input_data)

        prediction_minutes = float(prediction[0])

        # Prevent negative prediction
        prediction_minutes = max(
            0,
            prediction_minutes
        )

        rounded_prediction = round(
            prediction_minutes
        )


        # ====================================================
        # PREDICTION RESULT
        # ====================================================

        st.divider()

        st.subheader("🎯 Prediction Result")

        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                label="Estimated Delivery Time",
                value=f"{prediction_minutes:.1f} min"
            )


        with col2:

            st.metric(
                label="Approximate Time",
                value=f"{rounded_prediction} min"
            )


        st.success(
            f"🚚 Estimated delivery time: "
            f"**{rounded_prediction} minutes**"
        )


        # ====================================================
        # MODEL PERFORMANCE
        # ====================================================

        st.subheader("📊 Model Performance")

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                label="MAE",
                value="3.57 min"
            )


        with col2:

            st.metric(
                label="RMSE",
                value="4.55 min"
            )


        with col3:

            st.metric(
                label="R²",
                value="76.73%"
            )


        st.caption(
            "On the test dataset, the model's predictions "
            "are off by approximately 3.57 minutes on average."
        )


    except Exception as e:

        st.error(
            "An error occurred while making the prediction."
        )

        st.exception(e)


# ============================================================
# 8. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("ℹ️ About DeliveryETA")

    st.write(
        "DeliveryETA is a machine learning application "
        "that predicts delivery time in minutes."
    )

    st.write("**Final Model:** Optimized Random Forest")

    st.write("**Test MAE:** 3.57 minutes")

    st.write("**Test RMSE:** 4.55 minutes")

    st.write("**Test R²:** 0.7673")