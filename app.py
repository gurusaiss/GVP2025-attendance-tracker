import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pytesseract

# Set the Tesseract command path (adjust as needed)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

st.set_page_config(page_title="Zoom/Google Meet Attendance Tracker", layout="centered")
st.title("📷 Zoom/Google Meet Attendance Tracker")
st.markdown("Upload a screenshot of your Zoom or Google Meet class in **grid view**.")

uploaded_file = st.file_uploader("Upload Screenshot", type=["jpg", "jpeg", "png"])

def clean_name(name):
    """
    Remove unwanted characters, extra spaces and common OCR misreads.
    """
    name = name.strip().replace('\n', ' ')
    name = name.replace('|', 'I').replace('0', 'O').replace('1', 'I')
    name = ''.join(c for c in name if c.isalnum() or c.isspace())
    name = " ".join(name.split())
    name = name.title()
    if len(name) < 3 or name.isnumeric():
        return ""
    return name

def extract_text_from_image(image):
    """
    Use Tesseract's image_to_data to extract words with high confidence.
    """
    config = "--oem 3 --psm 6"
    data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
    words = []
    for i in range(len(data["text"])):
        try:
            conf = int(data["conf"][i])
        except:
            conf = 0
        if conf > 40 and data["text"][i].strip() != "":
            words.append(data["text"][i].strip())
    return " ".join(words)

def detect_attendance(image, rows, cols):
    """
    Process the image grid cell by cell:
      • Crop a region from the lower part of the cell.
      • Enhance and resize the region for clearer OCR.
      • Extract text and clean it to get the participant's full name.
      • Use face detection to assign attendance scores.
    """
    h, w, _ = image.shape
    grid_h, grid_w = h // rows, w // cols

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    attendance = {}

    for i in range(rows):
        for j in range(cols):
            # Define current cell boundaries
            x1, y1 = j * grid_w, i * grid_h
            x2, y2 = x1 + grid_w, y1 + grid_h
            cell = image[y1:y2, x1:x2]

            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            # Crop the lower part (starting at 60%) to capture the full name
            name_crop_y = int(grid_h * 0.60)
            name_area = gray[name_crop_y:, :]

            # Enhance contrast using CLAHE for better OCR readability
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(name_area)

            # Resize the region to double its size to aid in OCR accuracy
            resized = cv2.resize(enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

            # Use adaptive thresholding to get a clean binary image
            thresh = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)

            # Extract text using the image_to_data approach (aggregating high-confidence words)
            extracted_text = extract_text_from_image(thresh)
            name = clean_name(extracted_text)

            # Determine attendance points based on whether a face was detected.
            if not name:
                name = f"Unknown_{i}_{j}"
                attendance[name] = 0  # No name means absent
            else:
                attendance[name] = 1 if len(faces) > 0 else 0.5

    return attendance

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
        # Display the report in the format: Name : point
        for name, score in attendance.items():
            st.write(f"**{name}** : {score} points")

        # Prepare a downloadable CSV of the attendance
        df = pd.DataFrame(attendance.items(), columns=["Name", "Attendance Score"])
        st.download_button("📥 Download CSV", df.to_csv(index=False), "attendance.csv", "text/csv")