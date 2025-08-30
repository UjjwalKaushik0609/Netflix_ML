import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Load Model + Encoder
# -----------------------------
@st.cache_resource
def load_model():
    with open("netflix_rating_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le

model, le = load_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎬 Netflix Rating Prediction App")

st.write("This app predicts the **rating** (like TV-MA, PG-13, TV-14, etc.) "
         "for Netflix titles based on their features.")

# Collect inputs
type_ = st.selectbox("Type", ["Movie", "TV Show"])
director = st.text_input("Director (leave blank if unknown)", "")
cast = st.text_input("Main Cast (comma-separated, leave blank if unknown)", "")
country = st.text_input("Country", "United States")
release_year = st.number_input("Release Year", min_value=1920, max_value=2025, value=2020)
duration_num = st.number_input("Duration (in minutes or seasons)", min_value=1, value=90)
listed_in = st.text_input("Category (e.g. Drama, Comedy, Documentary)", "Drama")
year_added = st.number_input("Year Added to Netflix", min_value=2008, max_value=2025, value=2021)
is_international = st.selectbox("Is it International?", [0, 1])

# Prepare input DataFrame
input_data = pd.DataFrame([{
    "type": 0 if type_ == "Movie" else 1,
    "director": hash(director) % 5000,
    "cast": hash(cast) % 10000,
    "country": hash(country) % 1000,
    "release_year": release_year,
    "listed_in": hash(listed_in) % 500,
    "year_added": year_added,
    "duration_num": duration_num,
    "is_international": is_international
}])

# Predict
if st.button("Predict Rating"):
    pred = model.predict(input_data)[0]
    rating = le.inverse_transform([pred])[0]
    st.success(f"✅ Predicted Rating: **{rating}**")
