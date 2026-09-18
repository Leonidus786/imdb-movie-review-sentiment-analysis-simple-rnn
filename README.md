# IMDB Movie Review Sentiment Analysis Using Simple RNN

An end-to-end Natural Language Processing (NLP) and Deep Learning project that classifies movie reviews as **Positive** or **Negative** using a **Simple Recurrent Neural Network (RNN)**.

---

## Problem Statement

Develop an end-to-end Natural Language Processing and Deep Learning application that automatically classifies movie reviews as **Positive** or **Negative** using a **Simple Recurrent Neural Network (RNN)**.

The system takes a movie review in natural language as input, converts the text into numerical word representations using the IMDB vocabulary and an embedding layer, applies sequence padding to maintain a fixed input length, and passes the processed sequence through a Simple RNN-based deep learning model.

The trained model generates a sentiment prediction along with a prediction score. The model is then saved as an H5 file and integrated into a **Streamlit web application**, where users can enter their own movie reviews and receive a real-time sentiment prediction.

---

## Dataset

The project uses the **IMDB Movie Review Dataset** containing:

- **25,000** training reviews
- **25,000** testing reviews
- Binary sentiment labels:
  - `1` → Positive
  - `0` → Negative

---

## Input

A textual movie review, for example:

> "This movie was fantastic. The acting was excellent and the story was thrilling."

---

## Output

The application returns:

- **Sentiment:** Positive / Negative
- **Prediction Score:** Model prediction score

---

## Technical Workflow

```text
Movie Review
      ↓
Text Preprocessing
      ↓
Word Index / Integer Encoding
      ↓
Sequence Padding
      ↓
Embedding Layer
      ↓
Simple RNN
      ↓
Dense Layer + Sigmoid
      ↓
Sentiment Prediction
      ↓
Save Model as H5
      ↓
Streamlit Web Application
      ↓
Cloud Deployment
```
