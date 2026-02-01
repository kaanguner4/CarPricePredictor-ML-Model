import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

# --- PAGE SETTINGS ---
st.set_page_config(
    page_title="CatPricePredictor | Advanced Price Prediction",
    page_icon="🚀",
    layout="wide"  # Wide mode is better for more input fields
)

# --- LOAD THE MODEL ---
@st.cache_resource
def load_model():
    try:
        model = CatBoostRegressor()
        model.load_model("car_price_model.cbm")
        return model
    except Exception as e:
        st.error(f"Model could not be loaded! Error: {e}")
        return None

model = load_model()

st.title("🚗 CatPricePredictor: Professional Vehicle Valuation")
st.markdown(
    "Our model analyzes details such as color, accident history, and turbo features to produce the most realistic estimate."
)

# --- INPUT FIELDS ---
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📌 Basic Information")
        brand = st.text_input("Brand", value="Ford")
        model_name = st.text_input("Model", value="F-150")
        year = st.number_input("Model Year", 1990, 2026, 2020)
        mileage = st.number_input("Mileage (Miles)", 0, 500000, 50000)
        fuel = st.selectbox(
            "Fuel Type",
            ["Gasoline", "Hybrid", "E85 Flex Fuel", "Diesel", "Electric"]
        )

    with col2:
        st.subheader("🎨 Appearance & Condition")
        ext_col = st.text_input("Exterior Color", value="Black")
        int_col = st.text_input("Interior Color", value="Black")
        clean_title = st.selectbox("Clean Title", ["Yes", "No"])
        accident = st.selectbox(
            "Accident Status",
            ["None reported", "At least 1 accident reported"]
        )
        transmission = st.selectbox(
            "Transmission",
            ["automatic", "manual", "cvt"]
        )

    with col3:
        st.subheader("⚡ Technical Details")
        hp = st.number_input("Horsepower (HP)", 50, 3000, 250)
        liters = st.number_input("Engine Size (L)", 0.0, 20.0, 3.5)
        cylinders = st.number_input("Cylinders", 0, 20, 0)
        is_turbo = st.radio(
            "Is Turbo Available?",
            [1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    submit = st.form_submit_button(
        "💰 Calculate Price",
        use_container_width=True
    )

# --- PREDICTION LOGIC ---
if submit:
    if model:
        user_input = {
            'brand': brand,
            'model': model_name,
            'fuel_type': fuel,
            'ext_col': ext_col,
            'int_col': int_col,
            'clean_title': clean_title,
            'accident': accident,
            'transmission_type': transmission,
            'age': 2026 - year,
            'milage_num': float(mileage),
            'hp': float(hp),
            'liters': float(liters),
            'cylinders': float(cylinders),
            'is_turbo': int(is_turbo),
            'is_hybrid': 1 if "HYBRID" in fuel.upper() else 0
        }

        input_df = pd.DataFrame([user_input])

        # Make prediction
        res_log = model.predict(input_df)[0]
        final_price = np.expm1(res_log)

        # Result display
        st.divider()
        st.balloons()

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Estimated Value", f"${final_price:,.2f}")
        with c2:
            # Price range (Confidence interval using model MAE margin)
            st.info(
                f"Depending on market conditions, your vehicle value may range between "
                f"**${final_price*0.95:,.0f}** and **${final_price*1.05:,.0f}**."
            )
    else:
        st.error("Calculation could not be completed because the model file was not found.")

st.sidebar.markdown(
    "### About the Project\n"
    "This model is a **CatBoost** regressor algorithm trained on used car market data."
)
