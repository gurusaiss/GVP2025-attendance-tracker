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
from difflib import get_close_matches

# Optional: known list of student names
known_names = [
    "Vedha Priya", "Koushik Naramshetti", "Mounika Boja", "Harini Sree Koreti",
    "Harshini Pulugurtha", "Kavya Harshitha", "Syam Chagan Banisetti", 
    "Poorna Chandra Rao Pentakota", "Aditya Vanam", "Nikhil Ropate",
    "Ruchitha Kommineni", "Sujith Somi", "Tharun Gannamaneni", 
    "Gnann Saketh Periyala", "Bhargavi", "Pavani Keerthi", 
    "Yaswitha Alla", "Nanditha Donthamsetti", "Vardhan Gh", 
    "Sushmitha Boja", "Sumith Guru Sai", "Himasri Chavali", 
    "Vineela Malla", "Sai Varshini", "Srihas Monyam", 
    "Santhil Priya", "Arudravihnr Bommu", "Jaswanth Musineni", 
    "Madhusudhan Tadimeti", "Kaviyaswini Thuthika", "Hiranvika Yeminen"
]

def enhance_image(img):
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    img = cv2.bilateralFilter(img, 11, 17, 17)
    img = cv2.convertScaleAbs(img, alpha=1.5, beta=20)
    return img

def match_name(name, known_list):
    matches = get_close_matches(name, known_list, n=1, cutoff=0.5)
    return matches[0] if matches else name

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
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            # Extract lower part for name (last 25%)
            name_area = gray[int(grid_h * 0.75):, :]
            name_area = enhance_image(name_area)
            _, name_thresh = cv2.threshold(name_area, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            text = pytesseract.image_to_string(name_thresh, config="--psm 6").strip()
            name = clean_name(text)

            if not name or name.startswith("Unknown"):
                name = f"Unknown_{i}_{j}"
            else:
                name = match_name(name, known_names)

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
