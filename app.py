import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# --- Page Configuration ---
st.set_page_config(page_title="Real Estate Intelligence", layout="wide")

# --- Load Model and Metadata ---
@st.cache_resource
def load_assets():
    model = joblib.load('house_model.joblib')
    with open('model_metadata.json', 'r') as f:
        meta = json.load(f)
    return model, meta

model, meta = load_assets()

# --- App Header ---
st.title("🏡 King County Property Intelligence Engine")
st.markdown("### 90.3% Accurate Sales & Rental Forecasting")

# --- Sidebar Inputs (Universal Specs) ---
st.sidebar.header("Property Specifications")
sqft_living = st.sidebar.slider("Interior Living Space (sqft)", 500, 10,000, 2000)
sqft_basement = st.sidebar.slider("Basement Space (sqft)", 0, 5000, 0)
beds = st.sidebar.number_input("Bedrooms", 1, 10, 3)
baths = st.sidebar.slider("Bathrooms (Rounded for analysis)", 1.0, 8.0, 2.5, step=0.25)
floors = st.sidebar.radio("Floors", [1, 1.5, 2, 2.5, 3])
grade = st.sidebar.select_slider("Structural Grade (1-13)", options=list(range(1, 14)), value=7)
yr_built = st.sidebar.number_input("Year Built", 1900, 2024, 1995)
lat = st.sidebar.number_input("Latitude (e.g., 47.51)", value=47.5112, format="%.4f")
long = st.sidebar.number_input("Longitude (e.g., -122.25)", value=-122.2570, format="%.4f")

# --- Tabs for Dual-Persona Analysis ---
tab1, tab2 = st.tabs(["🏠 Homebuyer Valuation", "📈 Investor/Rental Analysis"])

# --- Core Inference Logic ---
def generate_prediction():
    # 1. Rounding and Feature Engineering
    house_age = 2026 - yr_built
    total_sqft = sqft_living + sqft_basement
    is_semi = 1 if 4 <= grade <= 10 else 0
    is_structured = 1 if grade >= 11 else 0
    
    # Rental metrics
    bed_bath_ratio = beds / (round(baths) + 0.1)
    space_density = sqft_living / (beds + 0.1)

    # 2. Build the exact input row (Order matters!)
    # Note: We provide median values for columns not captured in the UI
    input_data = {col: 0 for col in meta['features']}
    input_data.update({
        'bedrooms': beds,
        'bathrooms': round(baths),
        'sqft_living': sqft_living,
        'sqft_lot': 5000,
        'floors': round(float(floors)),
        'sqft_above': sqft_living,
        'sqft_basement': sqft_basement,
        'yr_built': yr_built,
        'lat': lat,
        'long': long,
        'sqft_living15': sqft_living,
        'sqft_lot15': 5000,
        'house_age': house_age,
        'total_sqft': total_sqft,
        'Grade_Semi-Structured': is_semi,
        'Grade_Structured': is_structured,
        'bed_bath_ratio': bed_bath_ratio,
        'space_density': space_density,
        'condition': 3
    })
    
    input_df = pd.DataFrame([input_data])[meta['features']]
    pred_log = model.predict(input_df)[0]
    market_value = np.expm1(pred_log)
    
    # Rent logic based on Grade
    rent_mult = 0.0065 if is_structured else 0.0055
    est_rent = market_value * rent_mult
    
    return market_value, est_rent, bed_bath_ratio, space_density

if st.sidebar.button("Generate Full Property Report"):
    val, rent, bbr, dens = generate_prediction()

    with tab1:
        st.metric("Estimated Market Value", f"${val:,.2f}", help="90.3% Accuracy Confidence")
        st.info(f"**Structural Classification:** {'Structured' if grade >= 11 else 'Semi-Structured' if grade >= 4 else 'UnStructured'}")
        
    with tab2:
        col1, col2 = st.columns(2)
        col1.metric("Suggested Monthly Rent", f"${rent:,.2f}")
        col2.metric("Rentability Score", "High" if bbr >= 1.2 else "Standard")
        st.write(f"**Space Efficiency:** {dens:.1f} sqft/bedroom")