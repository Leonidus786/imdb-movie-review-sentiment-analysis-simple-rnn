# # Step 1: Import Libraries and Load the Model
# from pathlib import Path
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.datasets import imdb
# from tensorflow.keras.preprocessing import sequence
# from tensorflow.keras.models import load_model

# # Load the IMDB dataset word index
# word_index = imdb.get_word_index()
# reverse_word_index = {value: key for key, value in word_index.items()}

# # Load the pre-trained model with ReLU activation

# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# MODEL_PATH = PROJECT_ROOT / "models" / "simple_rnn_imdb.h5"

# model = load_model(MODEL_PATH, compile=False)
# # Step 2: Helper Functions
# # Function to decode reviews
# def decode_review(encoded_review):
#     return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# # Function to preprocess user input
# def preprocess_text(text):
#     words = text.lower().split()
#     encoded_review = [word_index.get(word, 2) + 3 for word in words]
#     padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
#     return padded_review


# import streamlit as st
# ## streamlit app
# # Streamlit app
# st.title('IMDB Movie Review Sentiment Analysis')
# st.write('Enter a movie review to classify it as positive or negative.')

# # User input
# user_input = st.text_area('Movie Review')

# if st.button('Classify'):

#     preprocessed_input=preprocess_text(user_input)

#     ## MAke prediction
#     prediction=model.predict(preprocessed_input)
#     sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'

#     # Display the result
#     st.write(f'Sentiment: {sentiment}')
#     st.write(f'Prediction Score: {prediction[0][0]}')
# else:
#     st.write('Please enter a movie review.')

from pathlib import Path

import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences


# ============================================================
# Configuration
# ============================================================

MAX_LENGTH = 500

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "simple_rnn_imdb.h5"


# ============================================================
# Load trained model
# ============================================================

@st.cache_resource
def load_sentiment_model():
    """Load the trained Simple RNN model."""
    return load_model(MODEL_PATH, compile=False)


model = load_sentiment_model()


# ============================================================
# Load IMDB vocabulary
# ============================================================

word_index = imdb.get_word_index()


# ============================================================
# Preprocessing
# ============================================================

def preprocess_text(text: str):
    """Convert raw review text into padded IMDB token sequences."""

    words = text.lower().split()

    encoded_review = [
        word_index.get(word, 2) + 3
        for word in words
    ]

    padded_review = pad_sequences(
        [encoded_review],
        maxlen=MAX_LENGTH,
        padding="pre"
    )

    return padded_review


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(
    page_title="IMDB Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

st.title("IMDB Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review below to classify its sentiment "
    "as Positive or Negative."
)


user_input = st.text_area(
    "Movie Review",
    placeholder=(
        "Example: This movie was fantastic. "
        "The acting was excellent and the story was thrilling."
    ),
    height=180
)


# ============================================================
# Prediction
# ============================================================

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

        st.write(
            f"**Prediction Score:** {positive_probability:.4f}"
        )
