import streamlit as st
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@st.cache_resource
def load_text_model():
    """
    Load text AI detection model.
    """
    model_name = "roberta-base-openai-detector"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model


@st.cache_resource
def load_image_model():
    """
    Load image model.
    """
    model = models.efficientnet_b0(pretrained=True)
    model.eval()
    return model


def get_image_transform():
    """
    Image preprocessing transforms.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])