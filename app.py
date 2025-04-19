import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pytesseract
from difflib import get_close_matches

# Set tesseract path if needed (adjust based on OS)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# Optional: Known student names to match against
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

st.set_page_config(page_title="Zoom/Google Meet Attendance Tracker", layout="centered")
st.title("📷 Zoom/Google Meet Attendance Tracker")
st.markdown("Upload a screenshot of your Zoom or Google Meet class in **grid view**.")

uploaded_file = st.file_uploader("Upload Screenshot", type=["jpg", "jpeg", "png"])

# Cleaning and correction functions
def clean_name(name):
    name = name.strip().replace('\n', ' ')
    name = name.replace('|', 'I').replace('0', 'O').replace('1', 'I')
    name = ''.join(c for c in name if c.isalnum() or c.isspace())
    name = " ".join(name.split()).title()
    return name if len(name) > 2 and not name.isnumeric() else ""

def match_name(name, known_list):
    matches = get_close_matches(name, known_list, n=1, cutoff=0.5)
    return matches[0] if matches else name

def extract_text_from_image(image):
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

            # Detect face
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

            # Focus on bottom 20% of tile (name usually appears here)
            name_region = gray[int(grid_h * 0.80):, :]

            # Enhance OCR
            name_region = cv2.resize(name_region, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            name_region = cv2.bilateralFilter(name_region, 9, 75, 75)
            name_region = cv2.equalizeHist(name_region)

            text = extract_text_from_image(name_region)
            name = clean_name(text)
            name = match_name(name, known_names)

            if name:
                attendance[name] = 1 if len(faces) > 0 else 0.5
            else:
                attendance[f"Unknown_{i}_{j}"] = 0

    return attendance


# Main UI
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
            st.write(f"**{name}** : {score} points")

        df = pd.DataFrame(attendance.items(), columns=["Name", "Attendance Score"])
        st.download_button("📥 Download CSV", df.to_csv(index=False), "attendance.csv", "text/csv")
