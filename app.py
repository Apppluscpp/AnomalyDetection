import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import io

# === Utility functions ===

def preprocess_image(img, apply_clahe=True, convert_gray=False):
    img = cv2.resize(img, (224, 224))

    if convert_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if apply_clahe:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl,a,b))
        img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    img = img.astype(np.float32) / 255.0
    return img

def extract_morphology_features(image):
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    num_blobs = len(contours)
    blob_areas = [cv2.contourArea(cnt) for cnt in contours]
    total_blob_area = sum(blob_areas)
    avg_blob_area = np.mean(blob_areas) if blob_areas else 0
    max_blob_area = max(blob_areas) if blob_areas else 0
    blob_area_ratio = total_blob_area / (gray.shape[0] * gray.shape[1])

    return [num_blobs, avg_blob_area, max_blob_area, blob_area_ratio]

# === Streamlit app ===

st.title("Bottle Defect Detector")
st.write("Upload a bottle image to check if it's defective or proper.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Convert file to OpenCV format
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Show original image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    processed_img = preprocess_image(image_bgr)
    input_img = np.expand_dims(processed_img, axis=0)

    # Load model (assume saved as 'vgg16_bottle_classifier.h5')
    model = load_model('vgg16_finetuned.h5')

    # Predict
    prediction = model.predict(input_img)[0][0]
    label = "Defective" if prediction > 0.5 else "Proper"
    confidence = prediction if label == "Defective" else 1 - prediction

    st.subheader("Prediction Result:")
    st.write(f"**{label}** (Confidence: {confidence:.2%})")

    # Optional: Show morphology features
    features = extract_morphology_features(processed_img)
    st.write("Extracted Morphology Features:")
    st.json({
        "Blob Count": features[0],
        "Average Blob Area": round(features[1], 2),
        "Max Blob Area": round(features[2], 2),
        "Blob Area Ratio": round(features[3], 4)
    })
