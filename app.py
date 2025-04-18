import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pytesseract

# Set the Tesseract executable path for GitHub Codespaces
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

st.set_page_config(page_title="Zoom/Google Meet Attendance Tracker", layout="centered")
st.title("📷 Zoom/Google Meet Attendance Tracker")
st.markdown("Upload a screenshot of your Zoom or Google Meet class in **grid view**.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

def clean_name(name):
    """Clean the OCR text output to improve accuracy."""
    name = name.replace('|', 'I').replace('0', 'O').strip()  # Handle OCR errors
    return name

def preprocess_image(image):
    """Enhance image for better text recognition."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to make the text stand out
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Optionally, apply additional preprocessing like dilating or eroding
    # thresh = cv2.dilate(thresh, None, iterations=1)
    
    return thresh

def detect_attendance(image, rows, cols):
    """Detect attendance based on faces and names."""
    h, w, _ = image.shape
    grid_h, grid_w = h // rows, w // cols

    attendance = {}

    for i in range(rows):
        for j in range(cols):
            x1, y1 = j * grid_w, i * grid_h
            x2, y2 = x1 + grid_w, y1 + grid_h
            cell = image[y1:y2, x1:x2]

            # Preprocess the cell to enhance text for OCR
            preprocessed_cell = preprocess_image(cell)

            # Use Tesseract to extract text from the preprocessed image
            text = pytesseract.image_to_string(preprocessed_cell)
            name = clean_name(text.split("\n")[0]) if text.strip() else f"Unknown_{i}_{j}"

            # Use face detection to improve attendance determination
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            # Assign attendance score based on face detection
            if name and "Unknown" not in name:
                if len(faces) > 0:
                    attendance[name] = 1  # Full attendance if face is detected
                else:
                    attendance[name] = 0.5  # Video off if no face detected

    return attendance

if uploaded_file:
    image = Image.open(uploaded_file)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    st.image(image, caption="Uploaded Screenshot", use_column_width=True)

    rows = st.number_input("Number of Rows", min_value=1, max_value=10, value=3)
    cols = st.number_input("Number of Columns", min_value=1, max_value=10, value=3)

    if st.button("Process Attendance"):
        with st.spinner("Detecting..."):
            attendance = detect_attendance(image_cv, int(rows), int(cols))

        st.subheader("📊 Attendance Report")
        for name, score in attendance.items():
            st.write(f"**{name}**: {score} points")

        # Display results in a DataFrame and provide CSV download
        df = pd.DataFrame(attendance.items(), columns=["Name", "Attendance Score"])
        st.download_button("📥 Download CSV", df.to_csv(index=False), "attendance.csv", "text/csv")
