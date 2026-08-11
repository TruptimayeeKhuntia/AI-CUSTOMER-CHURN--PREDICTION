import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #eef2f3, #dfe9f3);
}

.main-container {
    max-width: 1100px;
    margin: auto;
    padding: 20px;
}

.header {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    padding: 35px;
    border-radius: 20px;
    text-align: center;
    color: white;
    margin-bottom: 30px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.15);
}

.header h1 {
    font-size: 42px;
    margin-bottom: 10px;
}

.header p {
    font-size: 18px;
    opacity: 0.9;
}

.input-card {
    background: white;
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.10);
    margin-bottom: 25px;
}

.result-card {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    color: white;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.15);
    margin-top: 25px;
}

.result-card h2 {
    font-size: 25px;
}

.result-card h1 {
    font-size: 45px;
    margin: 10px;
}

div.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 12px;
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    color: white;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #2a5298, #1e3c72);
    color: white;
}

.footer {
    text-align: center;
    margin-top: 40px;
    color: #555;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


try:
    model = joblib.load("model_dir/housing_model.joblib")

except Exception as e:
    st.error("Model could not be loaded. Check the model path.")
    st.stop()



st.markdown("""
<div class="header">
    <h1>🏠 House Price Prediction</h1>
    <p>Predict the median house value using Machine Learning</p>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div class="input-card">
    <h2>🏡 Enter House Details</h2>
    <p>Enter the details of the house below to predict its estimated value.</p>
</div>
""", unsafe_allow_html=True)


col1, col2 = st.columns(2)


with col1:

    longitude = st.number_input(
        "📍 Longitude",
        value=-122.23,
        format="%.4f"
    )

    latitude = st.number_input(
        "📍 Latitude",
        value=37.88,
        format="%.4f"
    )

    housing_median_age = st.number_input(
        "🏠 Housing Median Age",
        min_value=1.0,
        value=25.0
    )

    total_rooms = st.number_input(
        "🚪 Total Rooms",
        min_value=1.0,
        value=2000.0
    )


with col2:

    total_bedrooms = st.number_input(
        "🛏️ Total Bedrooms",
        min_value=1.0,
        value=400.0
    )

    population = st.number_input(
        "👨‍👩‍👧 Population",
        min_value=1.0,
        value=1000.0
    )

    households = st.number_input(
        "🏘️ Households",
        min_value=1.0,
        value=300.0
    )

    median_income = st.number_input(
        "💰 Median Income",
        min_value=0.0,
        value=4.0,
        format="%.2f"
    )



st.write("")

predict_button = st.button("🔮 Predict House Price")



if predict_button:

    new_house = pd.DataFrame({
        "longitude": [longitude],
        "latitude": [latitude],
        "housing_median_age": [housing_median_age],
        "total_rooms": [total_rooms],
        "total_bedrooms": [total_bedrooms],
        "population": [population],
        "households": [households],
        "median_income": [median_income]
    })

    try:

        # Model prediction in USD
        prediction = model.predict(new_house)

        predicted_price_usd = prediction[0]

        USD_TO_INR = 95.42

        predicted_price_inr = predicted_price_usd * USD_TO_INR

        st.markdown(f"""
        <div class="result-card">

        <h2>🏠 Estimated Median House Value</h2>

        <h1>₹{predicted_price_inr:,.0f}</h1>

        <p>Approximately ${predicted_price_usd:,.2f} USD</p>

        <p>💱 Conversion Rate: 1 USD = ₹{USD_TO_INR}</p>

        </div>
        """, unsafe_allow_html=True)

    except Exception as e:

        st.error(f"Prediction failed: {e}")

st.markdown("""
<div class="footer">
    <p>Built with ❤️ using Python, Machine Learning & Streamlit</p>
</div>
""", unsafe_allow_html=True)


