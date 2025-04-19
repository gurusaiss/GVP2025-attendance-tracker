import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pytesseract

# Specify Tesseract path (adjust if needed)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

st.set_page_config(page_title="Zoom/Google Meet Attendance Tracker", layout="centered")
st.title("📷 Zoom/Google Meet Attendance Tracker")
st.markdown("Upload a screenshot of your Zoom or Google Meet class in **grid view**.")

uploaded_file = st.file_uploader("Upload Screenshot", type=["jpg", "jpeg", "png"])

# Enhanced clean_name function to remove unwanted characters and extra spaces.
def clean_name(name):
    # Remove line breaks, extra spaces and unwanted characters
    name = name.strip().replace('\n', ' ')
    # Fix common OCR misinterpretations and convert to title case
    name = name.replace('|', 'I').replace('0', 'O').replace('1', 'I')
    name = ''.join(c for c in name if c.isalnum() or c.isspace())
    name = " ".join(name.split())
    name = name.title()
    # Very short text is likely garbage
    if len(name) < 3 or name.isnumeric():
        return ""
    return name

# Improved function to detect attendance in a grid image
def detect_attendance(image, rows, cols):
    h, w, _ = image.shape
    grid_h, grid_w = h // rows, w // cols

    # Adjust parameters for face detection if necessary
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    attendance = {}

    for i in range(rows):
        for j in range(cols):
            x1, y1 = j * grid_w, i * grid_h
            x2, y2 = x1 + grid_w, y1 + grid_h
            cell = image[y1:y2, x1:x2]

            # Convert to grayscale for both face detection and OCR
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

            # Detect faces in the cell
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            # Crop the lower portion where the name is expected (adjust if necessary)
            name_area = gray[int(grid_h * 0.65):, :]

            # Enlarge the region to help Tesseract read small texts
            name_area = cv2.resize(name_area, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            
            # Use adaptive threshold for a cleaner binary image
            name_thresh = cv2.adaptiveThreshold(name_area, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                cv2.THRESH_BINARY, 11, 2)

            # Optionally, a slight blur can reduce noise before OCR
            name_thresh = cv2.GaussianBlur(name_thresh, (3, 3), 0)

            # OCR: Using OEM 3 and PSM 7 (treat as a single text line)
            config = "--oem 3 --psm 7"
            text = pytesseract.image_to_string(name_thresh, config=config)
            name = clean_name(text)
            
            # Determine the attendance score:
            # 1 point if face detected indicating video on, 0.5 if no face (video off)
            if not name:
                # When no clear name is recognized, label as unknown with 0 points
                name = f"Unknown_{i}_{j}"
                attendance[name] = 0
            else:
                attendance[name] = 1 if len(faces) > 0 else 0.5

    return attendance

# Streamlit UI logic
if uploaded_file:
    image = Image.open(uploaded_file)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    st.image(image, caption="Uploaded Screenshot", use_container_width=True)

    # Input number of rows and columns based on grid view
    rows = st.number_input("Number of Rows", min_value=1, max_value=10, value=4)
    cols = st.number_input("Number of Columns", min_value=1, max_value=10, value=9)

    if st.button("Process Attendance"):
        with st.spinner("Detecting attendance and extracting names..."):
            attendance = detect_attendance(image_cv, int(rows), int(cols))

        st.subheader("📊 Attendance Report")
        # Display attendance report in Name : point format
        for name, score in attendance.items():
            st.write(f"**{name}** : {score} points")

        # Convert to CSV for download
        df = pd.DataFrame(attendance.items(), columns=["Name", "Attendance Score"])
        st.download_button("📥 Download CSV", df.to_csv(index=False), "attendance.csv", "text/csv")