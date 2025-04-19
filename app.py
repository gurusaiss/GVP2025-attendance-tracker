import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image
from difflib import get_close_matches

# Set up Tesseract command if needed
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Linux
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows

st.set_page_config(page_title="Zoom/Meet Attendance Tracker", layout="wide")
st.title("📸 Zoom/Google Meet Attendance Tracker")

# === Load Haar Cascade for face detection ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# === Known student names (extract from class list) ===
known_names = [
    "Vedha Priya", "Kaushik Naramsotti", "Mounica Kompella", "Harini Sree Koreti",
    "Harshini Pulugurtha", "Kusuma Harshitha", "Mounika Boja", "Syam Chaganisetti",
    "Poorna Chandra Rao Pentakota", "Hiranvika Yeminen", "Pavani Koerthi", 
    "Aashasvini Thuthika", "Ruchitha Kambhampati", "Sujith Soni", "Tharun Gannamaneni",
    "Gnan Seketh Periyala", "Bhargavi", "Sahanik Voluvati", "Yaswitha Alla", 
    "Nanditha Donthamsetti", "Vardhan Gh", "Sushmitha Boja", "Sumith Guru Sai", 
    "Himasri Chavali", "Vineela Malla", "Santhi Priya", "Arudra Vihari Bommma", 
    "Jayanth Musinana", "Srihas Manyam", "Aditya Varanasi", "Nikhil Rompte"
]

# === Helper Functions ===
def clean_text(text):
    return ''.join(c for c in text if c.isalnum() or c.isspace()).strip()

def match_name(raw_text):
    cleaned = clean_text(raw_text)
    matches = get_close_matches(cleaned, known_names, n=1, cutoff=0.6)
    return matches[0] if matches else None

def extract_faces_and_names(image):
    img_cv = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    
    h, w = gray.shape
    rows, cols = 5, 7  # Adjust based on layout
    cell_h, cell_w = h // rows, w // cols

    attendance = {}

    for i in range(rows):
        for j in range(cols):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            cell = gray[y1:y2, x1:x2]
            
            # Face detection
            faces = face_cascade.detectMultiScale(cell, scaleFactor=1.1, minNeighbors=3)

            # OCR
            text = pytesseract.image_to_string(cell, config="--psm 6")
            matched_name = match_name(text)

            # Attendance logic
            if matched_name:
                score = 1 if len(faces) > 0 else 0.5
                attendance[matched_name] = score
            else:
                attendance[f"Unknown_{i}_{j}"] = 0
    
    return attendance

# === Streamlit UI ===
uploaded_file = st.file_uploader("Upload screenshot of Zoom/Google Meet grid view", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Class Screenshot", use_column_width=True)

    with st.spinner("Analyzing attendance..."):
        attendance = extract_faces_and_names(image)

    st.subheader("📊 Attendance Report")
    for name, score in attendance.items():
        st.write(f"**{name}** : {score} points")

    # Optional: Download attendance as CSV
    import pandas as pd
    df = pd.DataFrame(attendance.items(), columns=["Name", "Attendance"])
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "attendance_report.csv", "text/csv")

