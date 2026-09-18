# Step 1: Import Libraries and Load the Model
from pathlib import Path

import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
MAX_LENGTH = 500
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "simple_rnn_imdb.h5"


# -------------------------------------------------------------------
# Load model
# -------------------------------------------------------------------
@st.cache_resource
def load_sentiment_model():
    """Load and cache the trained Simple RNN model."""
    return load_model(MODEL_PATH, compile=False)


model = load_sentiment_model()


# -------------------------------------------------------------------
# Load IMDB vocabulary
# -------------------------------------------------------------------
word_index = imdb.get_word_index()


# -------------------------------------------------------------------
# Preprocess user input
# -------------------------------------------------------------------
def preprocess_text(text: str):
    """Convert user text into the padded integer sequence expected by the model."""
    words = text.lower().split()

    encoded_review = [
        word_index.get(word, 2) + 3
        for word in words
    ]

    return pad_sequences(
        [encoded_review],
        maxlen=MAX_LENGTH,
        padding="pre"
    )


# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
st.set_page_config(
    page_title="IMDB Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

st.title("IMDB Movie Review Sentiment Analysis")
st.write(
    "Enter a movie review below to classify its sentiment as Positive or Negative."
)

user_input = st.text_area(
    "Movie Review",
    placeholder="Example: This movie was fantastic. The acting was excellent and the story was thrilling.",
    height=180
)

if st.button("Classify Sentiment", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a movie review.")
    else:
        preprocessed_input = preprocess_text(user_input)

        prediction = model.predict(
            preprocessed_input,
            verbose=0
        )

        positive_probability = float(prediction[0][0])
        sentiment = (
            "Positive"
            if positive_probability >= 0.5
            else "Negative"
        )

        st.subheader("Prediction")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Prediction Score:** {positive_probability:.4f}")
