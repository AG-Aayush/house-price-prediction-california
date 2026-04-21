import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ── 1. Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="California Housing Predictor",
    page_icon="🏠",
    layout="centered"
)

# ── 2. Load model ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load('housing_model.pkl')
    scaler = joblib.load('housing_scaler.pkl')
    return model, scaler

model, scaler = load_model()

# ── 3. Constants ───────────────────────────────────────────────
OCEAN_OPTIONS = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
AVG_HOUSEHOLDS_PER_BLOCK = 370
AVG_HOUSEHOLD_SIZE = 2.7
RMSE = 57265

# ── 4. Presets ─────────────────────────────────────────────────
PRESETS = {
    "Starter home":  dict(income_usd=25000, age=40, rooms=3, bedrooms=1, ocean='INLAND'),
    "Family home":   dict(income_usd=50000, age=25, rooms=6, bedrooms=3, ocean='<1H OCEAN'),
    "Luxury home":   dict(income_usd=100000, age=10, rooms=10, bedrooms=4, ocean='NEAR OCEAN'),
}

# ── 5. UI ──────────────────────────────────────────────────────
st.title("🏠 California House Price Predictor")

preset_name = st.selectbox("Start from preset", list(PRESETS.keys()))
p = PRESETS[preset_name]

col1, col2 = st.columns(2)

with col1:
    income_usd = st.number_input("Income ($)", 5000, 500000, int(p["income_usd"]))
    age = st.slider("House age", 1, 52, int(p["age"]))
    ocean = st.selectbox("Ocean proximity", OCEAN_OPTIONS, index=OCEAN_OPTIONS.index(p["ocean"]))

with col2:
    rooms = st.slider("Rooms", 1, 15, int(p["rooms"]))
    bedrooms = st.slider("Bedrooms", 1, 10, int(p["bedrooms"]))

# ── 6. Validation ──────────────────────────────────────────────
if bedrooms > rooms:
    st.warning("Bedrooms cannot exceed rooms")
    st.stop()

# ── 7. Prediction ──────────────────────────────────────────────
if st.button("Predict Price"):

    try:
        # Basic features
        income = float(income_usd) / 10000
        total_rooms = max(rooms * AVG_HOUSEHOLDS_PER_BLOCK, 1)
        total_bedrooms = max(bedrooms * AVG_HOUSEHOLDS_PER_BLOCK, 1)
        households = AVG_HOUSEHOLDS_PER_BLOCK
        population = households * AVG_HOUSEHOLD_SIZE * 1.1

        # Engineered features (must match training)
        age_depreciation = 1 - (age / 52) * 0.3
        bed_rooms_per_room = total_bedrooms / total_rooms
        income_age_adjusted = income * age_depreciation
        population_per_household = population / households
        rooms_per_household = total_rooms / households

        # Location (static)
        latitude = 35.63
        longitude = -119.57

        # Ocean encoding
        ocean_cols = {o: 0 for o in OCEAN_OPTIONS}
        ocean_cols[ocean] = 1

        # Build input dictionary
        input_dict = {
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
            "median_income": income,

            "age_depreciation": age_depreciation,
            "bed_rooms_per_room": bed_rooms_per_room,
            "income_age_adjusted": income_age_adjusted,
            "population_per_household": population_per_household,
            "rooms_per_household": rooms_per_household,

            "<1H OCEAN": ocean_cols["<1H OCEAN"],
            "INLAND": ocean_cols["INLAND"],
            "ISLAND": ocean_cols["ISLAND"],
            "NEAR BAY": ocean_cols["NEAR BAY"],
            "NEAR OCEAN": ocean_cols["NEAR OCEAN"],
        }

        # Convert to DataFrame
        input_data = pd.DataFrame([input_dict])

        # 🔥 CRITICAL: match training schema exactly
        for col in scaler.feature_names_in_:
            if col not in input_data:
                input_data[col] = 0

        input_data = input_data[scaler.feature_names_in_]

        # Predict
        scaled = scaler.transform(input_data)
        prediction = model.predict(scaled)[0]

        # Output
        st.success(f"Predicted Price: ${prediction:,.0f}")

        lower = prediction - RMSE
        upper = prediction + RMSE
        st.caption(f"Range: ${lower:,.0f} — ${upper:,.0f}")

        # Feature contribution (linear model)
        if hasattr(model, "coef_"):
            contributions = pd.Series(
                np.abs(model.coef_ * scaled[0]),
                index=scaler.feature_names_in_
            ).sort_values(ascending=False).head(6)

            st.subheader("Top Drivers")
            st.bar_chart(contributions)

    except Exception as e:
        st.error(f"Error: {e}")