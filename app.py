import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- HELPER FUNCTIONS ---
def format_inr(number):
    """Formats a number into Indian numbering system (Lakhs/Crores)"""
    if number >= 10000000:
        return f"₹{number / 10000000:.2f} Crores"
    elif number >= 100000:
        return f"₹{number / 100000:.2f} Lakhs"
    else:
        return f"₹{number:,.2f}"

def load_assets():
    try:
        with open('house_model_pack.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Error: 'house_model_pack.pkl' not found. Please run your training script first.")
        st.stop()

# --- LOAD MODEL PACK ---
assets = load_assets()
model = assets['model']
scaler = assets['scaler']
features = assets['features']

# --- APP CONFIG & UI ---
st.set_page_config(page_title="AI House Appraisal", page_icon="🏢", layout="centered")

st.title("🤖 Smart Real Estate AI Assistant")
st.markdown("""
Welcome! I am an AI trained on thousands of property records. 
Provide the details below, and I'll generate a market valuation for you in **Indian Rupees (INR)**.
""")

# --- INPUT SECTION ---
with st.container(border=True):
    st.subheader("📋 Property Specifications")
    col1, col2 = st.columns(2)
    
    with col1:
        area = st.number_input("Total Area (sqft)", min_value=500, max_value=20000, value=3000, step=100)
        bathrooms = st.select_slider("Number of Bathrooms", options=[1, 2, 3, 4, 5], value=2)
        bedrooms = st.select_slider("Number of Bedrooms", options=[1, 2, 3, 4, 5, 6], value=3)
        stories = st.selectbox("Number of Floors/Stories", [1, 2, 3, 4], index=0)

    with col2:
        parking = st.number_input("Parking Spaces", 0, 5, 1)
        ac = st.toggle("Air Conditioning Available", value=True)
        pref = st.toggle("Preferred Location/Area", value=False)
        guest = st.toggle("Guestroom Included", value=False)
        basement = st.toggle("Has Basement", value=False)

# --- PREDICTION LOGIC ---
if st.button("🚀 Generate AI Appraisal", use_container_width=True):
    # 1. Prepare data (must match training feature list exactly)
    # Note: We assume USD-based training data, converting to INR at the end
    USD_TO_INR = 83.50 
    
    input_dict = {
        'bedrooms': bedrooms, 
        'bathrooms': bathrooms, 
        'stories': stories,
        'mainroad': 1, 
        'guestroom': int(guest), 
        'basement': int(basement),
        'hotwaterheating': 0, 
        'airconditioning': int(ac), 
        'parking': parking, 
        'prefarea': int(pref),
        'area_sqrt': np.sqrt(area),
        'area_per_bath': np.sqrt(area) * bathrooms
    }
    
    # 2. Align features & Scale
    input_df = pd.DataFrame([input_dict]).reindex(columns=features, fill_value=0)
    input_scaled = scaler.transform(input_df)
    
    # 3. Predict & Invert Log
    log_price = model.predict(input_scaled)
    price_usd = np.expm1(log_price)[0]
    price_inr = price_usd * USD_TO_INR

    # --- THE LLM PERSONA OUTPUT ---
    st.divider()
    
    with st.chat_message("assistant"):
        st.write("### 📜 AI Market Appraisal Report")
        st.write(f"After processing your property specs through my XGBoost neural nodes, I've determined the estimated market value is:")
        
        # Highlighting the Price
        st.header(f"💰 {format_inr(price_inr)}")
        
        # Dynamic Commentary (Simulating LLM Insights)
        explanation = f"This valuation is heavily influenced by your **{area} sqft footprint**."
        if bathrooms >= 3:
            explanation += " Having 3+ bathrooms places this in a high-utility tier, significantly boosting the price."
        if ac:
            explanation += " The premium for Air Conditioning is factored in as a 'luxury' multiplier."
        if pref:
            explanation += " Since the property is in a **Preferred Area**, I have applied a location-based surge."
            
        st.info(explanation)
        
        st.caption(f"*Valuation based on current market data. Conversion rate used: 1 USD = ₹{USD_TO_INR}*")