import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pytesseract

# Tesseract path (for Codespaces or Linux environment)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

st.set_page_config(page_title="Zoom/Google Meet Attendance Tracker", layout="centered")
st.title("📷 Zoom/Google Meet Attendance Tracker")
st.markdown("Upload a screenshot of your Zoom or Google Meet class in **grid view**.")

uploaded_file = st.file_uploader("Upload Screenshot", type=["jpg", "jpeg", "png"])

# Clean the extracted name
def clean_name(name):
    name = name.strip()
    name = name.replace('|', 'I').replace('0', 'O').replace('1', 'I')
    name = ''.join(c for c in name if c.isalnum() or c.isspace())
    name = name.title()
    if len(name) < 3:
        return ""  # likely garbage
    return name

# Main attendance detector
def detect_attendance(image, rows, cols):
    h, w, _ = image.shape
    grid_h, grid_w = h // rows, w // cols

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    attendance = {}

    for i in range(rows):
        for j in range(cols):
            x1, y1 = j * grid_w, i * grid_h
            x2, y2 = x1 + grid_w, y1 + grid_h
            cell = image[y1:y2, x1:x2]

            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            # Extract name area (lower 35%)
            name_area = gray[int(grid_h * 0.65):, :]
            _, name_thresh = cv2.threshold(name_area, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # OCR to extract name
            text = pytesseract.image_to_string(name_thresh, config="--psm 7")
            name = clean_name(text)

            if not name:
                name = f"Unknown_{i}_{j}"
                attendance[name] = 0  # fully absent
            else:
                attendance[name] = 1 if len(faces) > 0 else 0.5

    return attendance

# UI logic
if uploaded_file:
    image = Image.open(uploaded_file)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    st.image(image, caption="Uploaded Screenshot", use_container_width=True)

    rows = st.number_input("Number of Rows", min_value=1, max_value=10, value=4)
    cols = st.number_input("Number of Columns", min_value=1, max_value=10, value=9)

    if st.button("Process Attendance"):
        with st.spinner("Detecting attendance and extracting names..."):
            attendance = detect_attendance(image_cv, int(rows), int(cols))

        st.subheader("📊 Attendance Report")
        for name, score in attendance.items():
            st.write(f"**{name}**: {score} points")

        df = pd.DataFrame(attendance.items(), columns=["Name", "Attendance Score"])
        st.download_button("📥 Download CSV", df.to_csv(index=False), "attendance.csv", "text/csv")
