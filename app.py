import streamlit as st
import pickle
from huggingface_hub import hf_hub_download

st.title("🎬 Netflix Rating Prediction App")

# Load Model + Label Encoder from Hugging Face Hub
@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(repo_id="UjjwalKaushik/Netflix_ML", filename="netflix_rating_model.pkl")
        encoder_path = hf_hub_download(repo_id="UjjwalKaushik/Netflix_ML", filename="label_encoder.pkl")

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(encoder_path, "rb") as f:
            le = pickle.load(f)

        return model, le
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

model, le = load_model()

if model is not None and le is not None:
    st.subheader("Enter Movie/Show Features")

    # --- User Input ---
    duration = st.number_input("Duration (minutes)", min_value=1, max_value=500, value=90)
    release_year = st.number_input("Release Year", min_value=1900, max_value=2025, value=2021)

    # Dropdown options
    country_options = ["United States", "India", "United Kingdom", "Canada", "Australia", "Other"]
    category_options = ["Documentaries", "Comedies", "Dramas", "Action & Adventure", "Kids", "Horror", "Other"]

    country = st.selectbox("Country", country_options)
    listed_in = st.selectbox("Category", category_options)

    # Dummy features (replace with your real preprocessing)
    features = [[0, 0, 0, 0, release_year, 0, 2021, duration, 0]]

    if st.button("Predict Rating"):
        pred = model.predict(features)
        # Handle label encoder safely
        if hasattr(le, "inverse_transform"):
            try:
                rating = le.inverse_transform(pred)[0]
            except Exception:
                rating = pred[0]
        else:
            rating = pred[0]
        st.success(f"✅ Predicted Rating: {rating}")
else:
    st.warning("⚠️ Model not loaded. Please check Hugging Face repo files.")

