import streamlit as st
import pandas as pd
import joblib

# ----------------------------------
# Load Trained Model
# ----------------------------------
model = joblib.load("carbon_footprint_model.pkl")

try:
    class_names = joblib.load("class_names.pkl")
except FileNotFoundError:
    class_names = None

# ----------------------------------
# Page Configuration
# ----------------------------------
st.set_page_config(
    page_title="Carbon Footprint Classification",
    page_icon="🌍",
    layout="wide"
)

# ----------------------------------
# Title
# ----------------------------------
st.title("🌍 Carbon Footprint Classification System")
st.write("Predict the carbon footprint class of a product.")

st.markdown("---")

# ----------------------------------
# Input Fields
# ----------------------------------

col1, col2 = st.columns(2)

with col1:

    product_type = st.selectbox(
        "Product Type",
        ["Electronics", "Clothing", "Furniture", "Food", "Appliances"]
    )

    material_composition = st.selectbox(
        "Material Composition",
        ["Plastic", "Metal", "Wood", "Glass", "Paper"]
    )

    weight_kg = st.number_input(
        "Weight (kg)",
        min_value=0.0,
        value=1.0
    )

    quantity = st.number_input(
        "Quantity",
        min_value=1,
        value=1
    )

    manufacturing_country = st.text_input(
        "Manufacturing Country",
        value="India"
    )

    transport_mode = st.selectbox(
        "Transport Mode",
        ["Road", "Rail", "Air", "Sea"]
    )

with col2:

    transport_distance_km = st.number_input(
        "Transport Distance (km)",
        min_value=0,
        value=100
    )

    packaging_type = st.selectbox(
        "Packaging Type",
        ["Box", "Bag", "Bottle", "Wrap"]
    )

    recycled_content_percent = st.slider(
        "Recycled Content (%)",
        0,
        100,
        20
    )

    renewable_energy_used_percent = st.slider(
        "Renewable Energy Used (%)",
        0,
        100,
        40
    )

    manufacturing_energy_kwh = st.number_input(
        "Manufacturing Energy (kWh)",
        min_value=0.0,
        value=50.0
    )

    packaging_weight_kg = st.number_input(
        "Packaging Weight (kg)",
        min_value=0.0,
        value=0.5
    )

    total_co2e_kg = st.number_input(
        "Total CO₂e (kg)",
        min_value=0.0,
        value=100.0
    )

st.markdown("---")

# ----------------------------------
# Prediction Button
# ----------------------------------

if st.button("Predict Carbon Class", type="primary"):

    input_data = pd.DataFrame({
        "product_type": [product_type],
        "material_composition": [material_composition],
        "weight_kg": [weight_kg],
        "quantity": [quantity],
        "manufacturing_country": [manufacturing_country],
        "transport_mode": [transport_mode],
        "transport_distance_km": [transport_distance_km],
        "packaging_type": [packaging_type],
        "recycled_content_percent": [recycled_content_percent],
        "renewable_energy_used_percent": [renewable_energy_used_percent],
        "manufacturing_energy_kwh": [manufacturing_energy_kwh],
        "packaging_weight_kg": [packaging_weight_kg],
        "total_co2e_kg": [total_co2e_kg],
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)

    st.success(f"Predicted Carbon Class: **{prediction}**")

    st.subheader("Prediction Probability")

    prob_df = pd.DataFrame(
        probability,
        columns=model.classes_
    )

    st.dataframe(prob_df, use_container_width=True)

    st.bar_chart(prob_df.T)

st.markdown("---")
st.caption("Developed using Streamlit and Machine Learning")
