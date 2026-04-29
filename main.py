import streamlit as st
from utils.text_analyzer import analyze_text
from utils.image_analyzer import analyze_image
from utils.video_analyzer import analyze_video

st.set_page_config(page_title="AI Multi-Source Integrity Guard", layout="wide")

st.title("🛡️ AI Multi-Source Integrity Guard")
st.subheader("Multimodal Fake Content Detector")

menu = st.sidebar.selectbox(
    "Choose Detection Mode",
    ["Text Detection", "Image Detection", "Video Detection"]
)

# TEXT
if menu == "Text Detection":
    st.header("📝 AI Text Detector")

    user_text = st.text_area("Enter text to analyze")

    if st.button("Analyze Text"):
        result = analyze_text(user_text)

        st.metric("Confidence Score", f"{result['confidence']}%")
        st.success(result["label"])
        st.info(result["report"])

# IMAGE
elif menu == "Image Detection":
    st.header("🖼️ Fake Image Detector")

    uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded:
        st.image(uploaded, width=400)

        if st.button("Analyze Image"):
            result = analyze_image(uploaded)

            st.metric("Confidence Score", f"{result['confidence']}%")
            st.success(result["label"])
            st.info(result["report"])

# VIDEO
elif menu == "Video Detection":
    st.header("🎥 Deepfake Video Detector")

    uploaded = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded:
        st.video(uploaded)

        if st.button("Analyze Video"):
            result = analyze_video(uploaded)

            st.metric("Confidence Score", f"{result['confidence']}%")
            st.success(result["label"])
            st.info(result["report"])