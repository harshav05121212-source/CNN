import cv2
import numpy as np
import streamlit as st
from PIL import Image
from huggingface_hub import snapshot_download
import onnxruntime as ort

st.set_page_config(page_title="Facial Emotion Recognition", page_icon="🎭", layout="centered")

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
MODEL_REPO = "dwest1507/emotion-detection-model"

@st.cache_resource
def load_model():
    folder = snapshot_download(
        repo_id=MODEL_REPO,
        allow_patterns=["emotion_classifier.onnx", "emotion_classifier.onnx.data"]
    )
    model_path = str(Path(folder) / "emotion_classifier.onnx")
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def detect_and_predict(image):
    image_rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return image_rgb, None

    session = load_model()
    input_name = session.get_inputs()[0].name
    output = image_rgb.copy()

    results = []

    # Process every detected face.
    for (x, y, w, h) in faces:
        face = image_rgb[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        face = face.astype(np.float32) / 255.0

        # ImageNet-style normalization used for EfficientNet models.
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        face = (face - mean) / std
        face = np.transpose(face, (2, 0, 1))
        face = np.expand_dims(face, axis=0).astype(np.float32)

        raw = session.run(None, {input_name: face})[0][0]
        probabilities = softmax(raw)
        index = int(np.argmax(probabilities))
        emotion = EMOTIONS[index]
        confidence = float(probabilities[index]) * 100

        results.append((emotion, confidence))
        label = f"{emotion} {confidence:.1f}%"

        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            output, label, (x, max(25, y-10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

    return output, results

st.title("🎭 Facial Emotion Recognition")
st.write("Upload a face image or take a camera snapshot. The CNN-based emotion model will detect the face and predict an emotion.")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
camera = st.camera_input("Or take a photo")

source = camera if camera is not None else uploaded

if source is not None:
    image = Image.open(source)
    result_image, results = detect_and_predict(image)

    st.image(result_image, caption="Detection Result", use_container_width=True)

    if results is None:
        st.warning("No face detected. Please try a clearer front-facing image.")
    else:
        st.subheader("Prediction")
        for i, (emotion, confidence) in enumerate(results, 1):
            st.success(f"Face {i}: **{emotion}** — {confidence:.1f}%")

st.divider()
st.caption("Educational mini-project. Emotion predictions are model outputs and should not be treated as definitive measurements of a person's internal emotional state.")
