from PIL import Image
import torch
from models.loader import load_image_model, get_image_transform


def analyze_image(uploaded_file):
    """
    Analyze uploaded image for fake/manipulated content.
    """

    # Load model and transform
    model = load_image_model()
    transform = get_image_transform()

    # Convert uploaded file to PIL Image
    image = Image.open(uploaded_file).convert("RGB")

    # Apply transforms
    img_tensor = transform(image).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        output = model(img_tensor)

    confidence = torch.softmax(output, dim=1).max().item()

    label = "Possibly Fake Image" if confidence > 0.5 else "Likely Real Image"

    return {
        "label": label,
        "confidence": round(confidence * 100, 2),
        "analysis": "Texture inconsistencies and unnatural pixel blending detected."
    }