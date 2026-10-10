# ============================================================
# FLOOD RISK CHECKER - Streamlit app
# Run with: streamlit run app.py   (after running flood_pipeline.py once)
# ============================================================

import streamlit as st  # Streamlit builds the web page
import pandas as pd  # Pandas builds the one-row table the model expects
import joblib  # loads the saved model
import os  # os helps us build file paths that work on any computer

st.set_page_config(page_title="Flood Risk Checker", page_icon="🌊")  # browser tab title and icon
st.title("🌊 Flood Risk Checker")  # big page title
st.write("Describe a 250m x 250m piece of land and the model estimates its flood risk.")  # short intro text


# build the path to the model file relative to THIS file, so it works on Colab, a laptop, and the cloud
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flood_model.joblib")  # folder of app.py + file name


@st.cache_resource  # load the model once and reuse it (faster app)
def load_model():  # function that loads the saved file
    return joblib.load(MODEL_PATH)  # returns the dict with "model" and "columns"


bundle = load_model()  # run the loader
model = bundle["model"]  # the trained model
columns = bundle["columns"]  # the exact column order the model was trained on

st.sidebar.header("Land details")  # heading for the input panel
elevation = st.sidebar.slider("Elevation (m)", 250, 450, 350)  # slider for height above sea level
slope = st.sidebar.slider("Slope (degrees)", 0.0, 25.0, 4.0)  # slider for ground steepness
dist_river = st.sidebar.slider("Distance to nearest river (km)", 0.05, 8.0, 1.5)  # slider for river distance
built_up = st.sidebar.slider("Built-up area (%)", 0, 100, 50)  # slider for how much is paved/built
rain_days = st.sidebar.slider("Heavy-rain days per year", 5, 40, 20)  # slider for rainfall
drainage = st.sidebar.selectbox("Drainage condition", ["good", "damaged", "blocked", "none"])  # dropdown for drainage

row = pd.DataFrame([{  # build a one-row table from the user's choices
    "elevation_m": elevation, "slope_deg": slope, "dist_river_km": dist_river,
    "built_up_pct": built_up, "heavy_rain_days": rain_days, "drainage_condition": drainage}])
row = pd.get_dummies(row, columns=["drainage_condition"])  # turn drainage text into 0/1 columns like in training
row = row.reindex(columns=columns, fill_value=0)  # match the training columns exactly (missing ones become 0)

if st.button("Check flood risk"):  # runs only when the button is clicked
    prob = model.predict_proba(row)[0][1]  # probability (0 to 1) that this land floods
    st.metric("Flood probability", f"{prob * 100:.1f}%")  # show the probability as a percentage
    if prob >= 0.6:  # high band
        st.error("🔴 HIGH risk - avoid building here without major drainage work.")  # red warning
    elif prob >= 0.35:  # medium band
        st.warning("🟠 MEDIUM risk - improve drainage and monitor after heavy rain.")  # orange warning
    else:  # low band
        st.success("🟢 LOW risk - flooding is unlikely under these conditions.")  # green message
    st.caption("Prototype trained on generated data - not for real planning decisions yet.")  # honest disclaimer
