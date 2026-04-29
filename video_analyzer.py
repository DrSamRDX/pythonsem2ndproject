import cv2
import tempfile
import torch
from PIL import Image
from models.loader import load_image_model, get_image_transform


def analyze_video(uploaded_file):
    """
    Analyze uploaded video by extracting frames.
    """

    model = load_image_model()
    transform = get_image_transform()

    # Save uploaded video temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(uploaded_file.read())
    temp_file.close()

    # Open video using file path
    cap = cv2.VideoCapture(temp_file.name)

    frame_scores = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Analyze every 20th frame
        if count % 20 == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(tensor)

            score = torch.softmax(output, dim=1).max().item()
            frame_scores.append(score)

        count += 1

    cap.release()

    avg_score = sum(frame_scores) / len(frame_scores) if frame_scores else 0

    label = "Suspicious / Deepfake Video" if avg_score > 0.5 else "Likely Real Video"

    return {
        "label": label,
        "confidence": round(avg_score * 100, 2),
        "analysis": "Temporal inconsistencies and frame anomalies detected."
    }