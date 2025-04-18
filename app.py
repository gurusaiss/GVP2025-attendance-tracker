import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


st.set_page_config(page_title="Zoom Attendance Tracker", layout="centered")

st.title("📷 Zoom Image Attendance Tracker")
st.markdown("Upload a screenshot of your Zoom class in grid view.")

uploaded_file = st.file_uploader(
    "Upload Zoom Image", type=["jpg", "jpeg", "png"])


def detect_faces_and_names(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5)

    attendance = {}

    # For each grid box (face + name region)
    for (x, y, w, h) in faces:
        # Expand below face box to include name area
        name_region_y_start = y + h
        name_region_y_end = y + h + int(h * 0.5)
        name_region = image[name_region_y_start:name_region_y_end, x:x + w]

        # OCR on cropped name region
        name_text = pytesseract.image_to_string(name_region)
        name_text = name_text.strip()

        if name_text:
            attendance[name_text] = 1  # Face + Name = Full attendance

    # OCR whole image to detect additional names (no face)
    full_text = pytesseract.image_to_string(gray)
    all_detected_names = [line.strip()
                          for line in full_text.split("\n") if line.strip()]

    for name in all_detected_names:
        if name not in attendance:
            attendance[name] = 0.5  # Name but no face = video off

    return attendance


def clean_name(name):
    name = name.replace('|', 'I')  # Fix OCR confusion
    name = name.replace('0', 'O')  # Zero to capital O
    return name.strip()

    name_text = clean_name(pytesseract.image_to_string(name_region).strip())


if uploaded_file:
    image = Image.open(uploaded_file)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    st.image(image, caption="Uploaded Zoom Screenshot", use_column_width=True)
    attendance = detect_faces_and_names(image_cv)

    st.subheader("📊 Attendance Report")
    for name, score in attendance.items():
        st.write(f"**{name}**: {score} points")
