import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go
import plotly.graph_objs as go


# -----------------------
# Load trained model
# -----------------------
with open("save_models/regModel.pkl,", "rb") as f:
    model = pickle.load(f)

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="Diabetes Prediction Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------
# Custom CSS for dark theme
# -----------------------
st.markdown("""
<style>
/* Background & font */
body {
    background-color: #1e1e2f;
    color: #e0e0e0;
    font-family: 'Segoe UI', sans-serif;
}

/* Title */
h1 {
    color: #00bcd4;
    text-align: center;
}

/* Card style for inputs */
.card {
    background-color: #2c2c3c;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 15px;
}

/* Button style */
.stButton>button {
    background-color: #00bcd4;
    color: #1e1e2f;
    font-weight: bold;
    border-radius: 8px;
    height: 45px;
    width: 180px;
}
.stButton>button:hover {
    background-color: #0097a7;
    color: #ffffff;
}

/* Footer */
footer {
    text-align: center;
    color: #00bcd4;
    font-size: 14px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------
# Title
# -----------------------
st.markdown("<h1>🩺 Diabetes Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Enter patient data to predict Diabetes probability</p>", unsafe_allow_html=True)
st.markdown("---")

# -----------------------
# Inputs in cards
# -----------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    pregnancies = st.slider("Pregnancies", 0, 20, 1)
    glucose = st.slider("Glucose", 0, 200, 120)
    blood_pressure = st.slider("Blood Pressure", 0, 140, 70)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    skin_thickness = st.slider("Skin Thickness", 0, 100, 20)
    insulin = st.slider("Insulin", 0, 900, 79)
    bmi = st.slider("BMI", 0.0, 70.0, 25.0, 0.1)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    diabetes_pedigree = st.slider("Diabetes Pedigree", 0.0, 2.5, 0.5, 0.01)
    age = st.slider("Age", 1, 120, 30)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# -----------------------
# Predict button
# -----------------------
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, diabetes_pedigree, age]])
    
    prediction = model.predict(input_data)[0]
    
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_data)[0][1]
    else:
        proba = 0

    # Prediction display
    if prediction == 1:
        st.markdown(f"<h2 style='color:#ff4b5c;'>❌ Diabetes Likely</h2>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h2 style='color:#4caf50;'>✅ Diabetes Unlikely</h2>", unsafe_allow_html=True)

    # Plotly gauge chart for probability
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba*100,
        title={'text': "Probability (%)"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "#00bcd4"},
               'steps': [
                   {'range': [0, 50], 'color': "#4caf50"},
                   {'range': [50, 100], 'color': "#ff4b5c"}]}
    ))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("<footer>Created by Md. Emon</footer>", unsafe_allow_html=True)
