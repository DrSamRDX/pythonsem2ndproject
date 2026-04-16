# AI Fake Content Detector (Streamlit GUI)

import streamlit as st
from transformers import pipeline

# Load model
@st.cache_resource
def load_model():
    model = pipeline("text-classification", model="roberta-base-openai-detector")
    return model

detector = load_model()

# UI Design
st.set_page_config(page_title="AI Fake Detector", layout="centered")

st.title("🧠 AI Fake Content Detector")
st.write("Detect whether a text is AI-generated or human-written.")

# Text input
user_input = st.text_area("✍️ Enter your text here:", height=200)

# Analyze button
if st.button("🔍 Analyze Text"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        result = detector(user_input)
        label = result[0]['label']
        score = result[0]['score']

        st.subheader("📊 Result")
        st.write(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {score:.2f}")

        if "FAKE" in label.upper():
            st.error("🚨 This content is likely AI-generated / Fake.")
        else:
            st.success("✅ This content appears Human-written.")

# File upload feature
st.subheader("📁 Upload a Text File")
uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")
    st.text_area("📄 File Content", text, height=200)

    if st.button("📊 Analyze File"):
        result = detector(text)
        label = result[0]['label']
        score = result[0]['score']

        st.write(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {score:.2f}")
