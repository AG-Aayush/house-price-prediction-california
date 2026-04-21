import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ── 1. Page config (must be first) ─────────────────────────────────────────────
st.set_page_config(
    page_title="California Housing Predictor",
    page_icon="🏠",
    layout="centered"
)

# ── 2. Background image function ───────────────────────────────────────────────
BG_IMAGES = {
    "None (custom)": "https://images.unsplash.com/photo-1568605114967-8130f3a36994?w=1400&q=80",
    "Starter home":  "https://images.unsplash.com/photo-1570129477492-45c003edd2be?w=1400&q=80",
    "Family home":   "https://images.unsplash.com/photo-1605276374104-dee2a0ed3cd6?w=1400&q=80",
    "Luxury home":   "https://images.unsplash.com/photo-1613977257363-707ba9348227?w=1400&q=80",
}

def set_background(url):
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.55)), url("{url}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .block-container {{ background: transparent !important; }}
    section[data-testid="stSidebar"] {{
        background: rgba(0,0,0,0.6) !important;
        backdrop-filter: blur(8px);
    }}
    div.stButton > button {{
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        color: white !important;
        border-radius: 10px;
    }}
    div.stButton > button:hover {{
        background: rgba(255,255,255,0.18) !important;
    }}
    div.stButton > button[kind="primary"] {{
        background: rgba(255,255,255,0.25) !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }}
    label, .stSlider label, p, h1, h2, h3, span, div {{
        color: white !important;
    }}
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {{
        background: rgba(255,255,255,0.1) !important;
        border-color: rgba(255,255,255,0.25) !important;
        color: white !important;
        border-radius: 8px !important;
    }}
    .stSuccess {{ background: rgba(0,180,100,0.25) !important; border-radius: 10px; }}
    .stWarning {{ background: rgba(255,180,0,0.25) !important; border-radius: 10px; }}
    </style>
    """, unsafe_allow_html=True)

# ── 3. Load model and scaler ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load('housing_model.pkl')
    scaler = joblib.load('housing_scaler.pkl')
    return model, scaler

model, scaler = load_model()
features      = scaler.feature_names_in_

# ── 4. Constants ───────────────────────────────────────────────────────────────
OCEAN_OPTIONS            = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
AVG_HOUSEHOLDS_PER_BLOCK = 370
AVG_HOUSEHOLD_SIZE       = 2.7
RMSE                     = 57265   # replace with your actual training RMSE

# ── 5. Presets ─────────────────────────────────────────────────────────────────
PRESETS = {
    "None (custom)": dict(income_usd=35_000,  age=20, rooms=5,  bedrooms=2, ocean='<1H OCEAN'),
    "Starter home":  dict(income_usd=25_000,  age=40, rooms=3,  bedrooms=1, ocean='INLAND'),
    "Family home":   dict(income_usd=50_000,  age=25, rooms=6,  bedrooms=3, ocean='<1H OCEAN'),
    "Luxury home":   dict(income_usd=100_000, age=10, rooms=10, bedrooms=4, ocean='NEAR OCEAN'),
}

# ── 6. Session state ───────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── 7. Sidebar — prediction history ───────────────────────────────────────────
with st.sidebar:
    st.header("Prediction history")
    if st.session_state.history:
        for h in reversed(st.session_state.history[-5:]):
            st.metric(h["label"], f"${h['price']:,.0f}")
            st.caption(f"Income ${h['income_usd']:,.0f} · Age {h['age']} yrs · {h['ocean']}")
            st.divider()
    else:
        st.caption("No predictions yet.")

# ── 8. Title ───────────────────────────────────────────────────────────────────
st.title("🏠 California House Price Predictor")
st.write("Choose a preset or enter your own values to predict house price.")

# ── 9. Preset selector ─────────────────────────────────────────────────────────
preset_name = st.selectbox("Start from a preset", list(PRESETS.keys()))
p           = PRESETS[preset_name]   # always defined before any widget reads it

set_background(BG_IMAGES[preset_name])

st.divider()

# ── 10. Input widgets ──────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    income_usd = st.number_input(
        "Median household income ($)",
        min_value=5_000,
        max_value=500_000,
        value=int(p["income_usd"]),
        step=5_000,
        help="Enter the annual household income in dollars"
    )
    age = st.slider(
        "House age (years)",
        1, 52, int(p["age"]),
        help="Newer houses score higher, older houses are discounted"
    )
    ocean = st.selectbox(
        "Ocean proximity",
        OCEAN_OPTIONS,
        index=OCEAN_OPTIONS.index(p["ocean"])
    )

with col2:
    rooms = st.slider(
        "Rooms in your house",
        1, 15, int(p["rooms"])
    )
    bedrooms = st.slider(
        "Bedrooms in your house",
        1, 10, int(p["bedrooms"])
    )

# ── 11. Validation ─────────────────────────────────────────────────────────────
if bedrooms > rooms:
    st.warning("Bedrooms can't exceed total rooms.")
    st.stop()

# ── 12. Predict button ─────────────────────────────────────────────────────────
if st.button("Predict Price", type="primary", use_container_width=True):

    income         = income_usd / 10_000
    total_rooms    = rooms    * AVG_HOUSEHOLDS_PER_BLOCK
    total_bedrooms = bedrooms * AVG_HOUSEHOLDS_PER_BLOCK
    households     = AVG_HOUSEHOLDS_PER_BLOCK
    population     = households * AVG_HOUSEHOLD_SIZE * 1.1

    inverted_age        = 53 - age
    age_depreciation    = (age / 52) * 0.3
    income_age_adjusted = income * (1 - age_depreciation)

    latitude  = 35.63
    longitude = -119.57

    ocean_cols = {o: 0 for o in OCEAN_OPTIONS}
    ocean_cols[ocean] = 1

    input_data = pd.DataFrame([[
        longitude,
        latitude,
        inverted_age,
        total_rooms,
        total_bedrooms,
        population,
        households,
        income_age_adjusted,
        ocean_cols["<1H OCEAN"],
        ocean_cols["INLAND"],
        ocean_cols["ISLAND"],
        ocean_cols["NEAR BAY"],
        ocean_cols["NEAR OCEAN"],
    ]], columns=features)

    scaled     = scaler.transform(input_data)
    prediction = model.predict(scaled)[0]

    lower = prediction - RMSE
    upper = prediction + RMSE

    st.session_state.history.append({
        "label":      preset_name if preset_name != "None (custom)" else "Custom",
        "price":      prediction,
        "income_usd": income_usd,
        "age":        age,
        "ocean":      ocean,
    })

    st.divider()
    st.success(f"### Predicted value: ${prediction:,.0f}")
    st.caption(f"Confidence range: ${lower:,.0f} – ${upper:,.0f}")

    penalty_pct = age_depreciation * 100
    st.info(f"Age adjustment applied: -{penalty_pct:.1f}% on income signal "
            f"({'new build' if age < 10 else 'modern' if age < 25 else 'older property'})")

    contributions = pd.Series(
        np.abs(model.coef_ * scaled[0]),
        index=features
    ).sort_values(ascending=False).head(6)

    st.subheader("What drove this prediction")
    st.bar_chart(contributions)