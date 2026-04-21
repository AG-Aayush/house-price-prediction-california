# House Price Prediction Web App (California Dataset)

## Project Overview

This project is a **Machine Learning-based House Price Prediction System** built using the California housing dataset. It predicts house prices based on multiple features such as income, location, population, and housing characteristics.

The project includes:

* Data preprocessing
* Exploratory Data Analysis (EDA)
* Model training using Linear Regression
* Model serialization (`.pkl` files)
* Web application for prediction

---

## Objective

To build a regression model that can accurately predict median house prices in California districts based on given input features.

---

##  Dataset

* Source: California Housing Dataset
* Features include:

  * Median Income
  * House Age
  * Average Rooms
  * Average Bedrooms
  * Population
  * Latitude & Longitude
* Target:

  * Median House Value

---

## Machine Learning Workflow

### 1. Data Preprocessing

* Handling missing values (if any)
* Feature scaling using StandardScaler
* Outlier detection and treatment (IQR method)

### 2. Exploratory Data Analysis (EDA)

* Correlation analysis
* Distribution plots
* Outlier visualization

### 3. Model Training

* Algorithm used: **Linear Regression**
* Train-test split applied
* Model evaluation using:

  * RMSE
  * R² Score

### 4. Model Saving

* Trained model saved as:

  * `housing_model.pkl`
* Scaler saved as:

  * `housing_scaler.pkl`

---

##  Results

* The model provides reasonable predictions based on linear relationships in the dataset.
* Visualization plots included for analysis:

  * Outlier detection plots
  * Regression fit plots
  * Residual analysis

---

## Web Application

A simple web interface is created using Python (Streamlit/Flask depending on implementation).

### Features:

* User inputs housing parameters
* Predicts house price instantly
* Uses trained ML model backend

---

## Project Structure

```
House_price_prediction_system/
│── linear_regression_app.py      # Web app
│── linear_regression.ipynb       # Model training notebook
│── housing.csv                   # Dataset
│── districts.csv                 # Additional dataset
│── housing_model.pkl             # Trained model
│── housing_scaler.pkl            # Feature scaler
│── requirements.txt              # Dependencies
│── *.png                         # Visualization plots
│── environment/                  # Virtual environment (not uploaded)
```

---

## Installation & Setup

### 1. Clone repository

```bash
git clone https://github.com/your-username/house-price-prediction-california.git
cd house-price-prediction-california
```

### 2. Create virtual environment

```bash
python -m venv environment
```

### 3. Activate environment

```bash
environment\Scripts\activate   # Windows
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run application

```bash
streamlit run linear_regression_app.py
```

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
streamlit
```

---

## Key Learnings

* Data preprocessing techniques
* Linear regression implementation
* Feature scaling importance
* Model persistence using pickle
* Basic web integration for ML models



## Author

**Aayush**
Engineering Student | ML & DevOps Enthusiast

---

## License

This project is for educational purposes.

