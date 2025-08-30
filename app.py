import streamlit as st
import pickle
import os

st.title("🎬 Netflix Rating Prediction")

# Debug: show files available in the repo
st.write("📂 Files available in repo:", os.listdir("."))

@st.cache_resource
def load_model():
    try:
        with open("netflix_rating_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
        return model, le
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}")
        return None, None

model, le = load_model()

if model is not None and le is not None:
    st.success("✅ Model and Label Encoder loaded successfully!")
    
    # Example input (you can replace with your cleaned features later)
    type_input = st.selectbox("Select Type", ["Movie", "TV Show"])
    release_year = st.number_input("Release Year", min_value=1920, max_value=2025, value=2020)
    duration = st.number_input("Duration (minutes)", min_value=1, max_value=300, value=90)
    is_international = st.selectbox("Is International?", [0, 1])

    if st.button("Predict Rating"):
        # Prepare input vector (this must match your training features order!)
        input_data = [[
            0 if type_input == "Movie" else 1,
            release_year,
            duration,
            is_international
        ]]
        prediction = model.predict(input_data)
        rating = le.inverse_transform(prediction)[0]
        st.success(f"📺 Predicted Rating: **{rating}**")
else:
    st.warning("⚠️ Model not loaded. Please check if `.pkl` files are in repo.")

