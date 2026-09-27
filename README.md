# 🚚 DeliveryETA

A Streamlit application that uses Machine Learning to predict delivery time.

Users enter information about the delivery person, order, weather, and traffic. The model then estimates the delivery duration in minutes.

## Features

- Enter delivery information.
- Consider delivery distance and pickup delay.
- Consider weather, traffic, and vehicle type.
- Predict delivery time.
- Display model performance metrics.
- Visualize delivery time distribution.

## Dataset

The dataset is available at:

```text
data/selected_features_data.csv
```

The main features include:

- Delivery person's age and rating
- Number of multiple deliveries
- Distance in kilometers
- Weather conditions
- Road traffic density
- Vehicle type
- Day of the week
- Order hour
- Pickup hour
- Pickup delay
- Month

The target variable is:

```text
Time_taken(min)
```

## Methodology

### Data Preparation

- Selection of relevant features.
- Conversion of month names into numerical values.
- Scaling of numerical features.
- One-Hot Encoding of categorical features.
- Preservation of the exact feature order used during training.

### Model

The model used is a `Random Forest Regressor`, optimized with `RandomizedSearchCV`.

The model files are stored in:

```text
models/delivery_eta_pipeline.joblib
models/scaler.joblib
```

## Results and Metrics

| Metric | Result |
|---|---:|
| MAE | 3.57 minutes |
| RMSE | 4.55 minutes |
| R² | 76.73% |

### Interpretation

- **MAE**: The average prediction error is approximately 3.57 minutes.
- **RMSE**: The root mean squared error is 4.55 minutes.
- **R²**: The model explains 76.73% of the variance in delivery times.

## Technologies

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib

## Installation

### Requirements

- Python 3.9 or higher
- Git

### Clone the repository

```bash
git clone <REPOSITORY_URL>
cd DeliveryETA
```

### Create a virtual environment

On Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install streamlit pandas numpy joblib matplotlib scikit-learn
```

## Run the Application

From the project root directory:

```bash
streamlit run app/app.py
```

The application will be available at:

```text
http://localhost:8501
```

## Project Structure

```text
DeliveryETA/
├── app/
│   └── app.py
├── data/
│   └── selected_features_data.csv
├── models/
│   ├── delivery_eta_pipeline.joblib
│   └── scaler.joblib
├── screenshots/
│   ├── prediction.png
│   ├── performance.png
│   └── distribution.png
└── README.md
```

## Screenshots

### Prediction Form

![Prediction Form](screenshots/prediction.png)

### Model Performance

![Model Performance](screenshots/performance.png)

### Delivery Time Distribution

![Delivery Time Distribution](screenshots/distribution.png)

> Add the screenshots to the `screenshots/` directory using the filenames shown above.