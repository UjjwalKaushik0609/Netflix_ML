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
    # --- User Input ---
    st.subheader("Enter Movie/Show Features")

    duration = st.number_input("Duration (minutes)", min_value=1, max_value=500, value=90)
    release_year = st.number_input("Release Year", min_value=1900, max_value=2025, value=2021)
    country = st.text_input("Country", "United States")
    listed_in = st.text_input("Category", "Documentaries")

    # Dummy encoding (you’d replace this with your real preprocessing)
    features = [[0, 0, 0, 0, release_year, 0, 2021, duration, 0]]

    if st.button("Predict Rating"):
        pred = model.predict(features)
        st.success(f"✅ Predicted Rating: {le.inverse_transform(pred)[0]}")
else:
    st.warning("⚠️ Model not loaded. Please check Hugging Face repo files.")

