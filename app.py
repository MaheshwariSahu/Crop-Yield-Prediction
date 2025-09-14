import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

# Load dataset
df = pd.read_csv("yield_df.csv")

# Page config
st.set_page_config(page_title="Crop Yield Predictor", layout="wide")

# Load model and preprocessor
model = pickle.load(open("model.pkl", "rb"))
preprocessor = pickle.load(open("preprocessor.pkl", "rb"))

# Sidebar Layout
with st.sidebar:
    st.markdown("## 🌾 Crop Yield Predictor")

    # Navigation buttons
    if st.button("🏠 Home"):
        st.session_state["page"] = "home"
    if st.button("🔮 Prediction"):
        st.session_state["page"] = "prediction"
    if st.button("📋 Input Summary"):
        st.session_state["page"] = "summary"
    if st.button("📊 Yield Trend"):
        st.session_state["page"] = "trend"

# Default page
if "page" not in st.session_state:
    st.session_state["page"] = "home"

# ---------- HOME ----------
if st.session_state["page"] == "home":
    st.title("🌾 Crop Yield Prediction")
    st.markdown("""
    Welcome to the **AI-powered Crop Yield Prediction System**.  
    Use this tool to:
    - Predict crop yield based on rainfall, pesticides, temperature, and region.  
    - View a summary of your inputs.  
    - Explore simulated yield trends with interactive graphs.  

    👉 Navigate to the **Prediction** tab from the sidebar to start.
    """)

# ---------- PREDICTION ----------
elif st.session_state["page"] == "prediction":
    st.header("🔮 Enter Input Parameters")

    # --- Extract categories from preprocessor ---
    area_categories = ["India"]  # fallback
    item_categories = ["Rice"]   # fallback

    try:
        for name, transformer, cols in preprocessor.transformers_:
            if isinstance(transformer, OneHotEncoder):
                cats = transformer.categories_
                if len(cats) >= 2:  # Area + Item both present
                    area_categories = cats[0].tolist()
                    item_categories = cats[1].tolist()
    except Exception as e:
        st.warning(f"⚠️ Could not extract categories from preprocessor. Using defaults. ({e})")

    col1, col2 = st.columns(2)

    with col1:
        year = st.number_input("Year", min_value=1900, max_value=2100, value=1990, step=1)
        rainfall = st.number_input("Average Rainfall (mm per year)", value=1485.0, step=10.0)
        pesticides = st.number_input("Pesticides (tonnes)", value=121.0, step=1.0)

    with col2:
        temperature = st.number_input("Average Temperature (°C)", value=16.37, step=0.1)
        area = st.selectbox("Area (Country/Region)", area_categories)
        crop = st.selectbox("Crop (Item)", item_categories)

    # Prediction button
    if st.button("🚀 Predict Crop Yield"):
        input_df = pd.DataFrame(
            [[year, rainfall, pesticides, temperature, area, crop]],
            columns=['Year','average_rain_fall_mm_per_year','pesticides_tonnes','avg_temp','Area','Item']
        )

        # Apply preprocessing
        processed_input = preprocessor.transform(input_df)
                
        # Store inputs & prediction in session state
        st.session_state["inputs"] = {
            "Year": year,
            "Rainfall": rainfall,
            "Pesticides": pesticides,
            "Temperature": temperature,
            "Area": area,
            "Crop": crop
        }
       
        # Prediction
        prediction = model.predict(processed_input)
        st.session_state["prediction"] = prediction
        st.success(f"🌱 Predicted Crop Yield: **{prediction[0]:.2f} tonnes/hectare**")

# ---------- INPUT SUMMARY ----------
elif st.session_state["page"] == "summary":
    st.header("📋 Input Summary")
    if "inputs" in st.session_state:
        # Convert dictionary to two-column table
        input_df = pd.DataFrame(
            list(st.session_state["inputs"].items()), 
            columns=["Parameter", "Value"]
        )
        st.table(input_df)
    else:
        st.warning("No inputs available. Please make a prediction first.")

# ---------- YIELD TREND ----------
elif st.session_state["page"] == "trend":
    st.header("📊 Yield Trend Based on Inputs")

    if "prediction" in st.session_state and "inputs" in st.session_state:
        # Filter dataset for selected Area & Crop
        area = st.session_state["inputs"]["Area"]
        crop = st.session_state["inputs"]["Crop"]

        trend_df = df[(df["Area"] == area) & (df["Item"] == crop)]

        if not trend_df.empty:
            # Sort by year
            trend_df = trend_df.sort_values("Year")

            fig, ax = plt.subplots()
            ax.plot(trend_df["Year"], trend_df["hg/ha_yield"], marker="o", color="green")

            ax.set_xlabel("Year")
            ax.set_ylabel("Actual Yield (hg/ha)")
            ax.set_title(f"Yield Trend for {crop} in {area}")

            st.pyplot(fig)
        else:
            st.warning("⚠️ No historical data available for this Area and Crop.")
    else:
        st.warning("⚠️ Please make a prediction first.")

