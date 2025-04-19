import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="Attendance Tracker", layout="wide")
st.title("📸 Zoom/Google Meet Attendance Tracker")

uploaded_file = st.file_uploader("Upload class screenshot (Zoom/Meet grid view)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load and preprocess image
    image = cv2.imread(tmp_path)
    image = cv2.resize(image, (1600, 900))  # resize to ensure consistency
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 8)

    # OCR configuration
    custom_config = r'--oem 3 --psm 6'
    data = pytesseract.image_to_data(thresh, config=custom_config, output_type=pytesseract.Output.DICT)

    # Collect detected names
    names_detected = []
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        conf = int(data['conf'][i])
        if conf > 60 and len(text) > 2 and text.lower() not in ["mic", "video"]:
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            names_detected.append((text, (x, y, w, h)))
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Show annotated image
    st.image(image, caption="Detected Names", use_column_width=True)

    # Display attendance
    st.subheader("📊 Attendance Report")
    if names_detected:
        for name, _ in names_detected:
            st.write(f"✅ {name} : 1 point")
    else:
        st.warning("⚠️ No names detected. Please upload a clearer screenshot.")
