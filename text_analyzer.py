"""
Text Analyzer Module

Handles AI-generated text detection logic.
"""

import torch
from models.loader import load_text_model


def analyze_text(text: str) -> dict:
    """
    Analyze text to determine likelihood of AI generation.

    Args:
        text (str): Input text

    Returns:
        dict: Analysis results including score and explanation
    """
    tokenizer, model = load_text_model()

    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    ai_score = probs[0][1].item()

    explanation = generate_text_explanation(text, ai_score)

    return {
        "confidence": ai_score,
        "label": "AI-Generated" if ai_score > 0.5 else "Human-Written",
        "analysis": explanation
    }


def generate_text_explanation(text: str, score: float) -> str:
    """
    Generate human-readable explanation.

    Args:
        text (str): Input text
        score (float): AI probability

    Returns:
        str: Explanation string
    """
    if score > 0.7:
        return "The text shows repetitive patterns and lacks stylistic variation, indicating synthetic generation."
    elif score > 0.5:
        return "Some linguistic patterns suggest AI involvement, but not strongly conclusive."
    else:
        return "Text appears natural with human-like variation and structure."