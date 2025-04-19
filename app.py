import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pytesseract
import re

# Tesseract path (for Codespaces or Linux environment)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

st.set_page_config(page_title="Zoom/Google Meet Attendance Tracker", layout="centered")
st.title("📷 Zoom/Google Meet Attendance Tracker")
st.markdown("Upload a screenshot of your Zoom or Google Meet class in **grid view**.")

uploaded_file = st.file_uploader("Upload Screenshot", type=["jpg", "jpeg", "png"])

def clean_name(name):
    # Remove common OCR artifacts and normalize
    name = name.strip()
    
    # Common OCR corrections
    corrections = {
        '|': 'I',
        '0': 'O',
        '1': 'I',
        '5': 'S',
        '8': 'B',
        '[': '',
        ']': '',
        '(': '',
        ')': '',
        '{': '',
        '}': '',
        '<': '',
        '>': '',
        '*': '',
        '#': '',
        '@': '',
        '!': '',
        '?': '',
        '~': '',
        '`': '',
        '^': '',
        '&': '',
        '_': '-',
        ';': '',
        ':': '',
        ',': '',
        '.': '',
        '"': '',
        "'": ''
    }
    
    for wrong, right in corrections.items():
        name = name.replace(wrong, right)
    
    # Remove multiple spaces and normalize case
    name = re.sub(r'\s+', ' ', name).strip()
    name = name.title()
    
    # Filter out very short names (likely garbage)
    if len(name) < 3:
        return ""
    
    # Remove any remaining non-alphabetic characters (keeping spaces and hyphens)
    name = re.sub(r'[^a-zA-Z\s\-]', '', name)
    
    return name

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

            # Convert to grayscale for processing
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            
            # Enhance contrast for better OCR
            gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)

            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            # Extract name area (lower 20-30% of the cell)
            name_area = gray[int(grid_h * 0.7):int(grid_h * 0.9), int(grid_w * 0.1):int(grid_w * 0.9)]
            
            # Thresholding for better text extraction
            _, name_thresh = cv2.threshold(name_area, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Apply dilation to make text more solid
            kernel = np.ones((2, 2), np.uint8)
            name_thresh = cv2.dilate(name_thresh, kernel, iterations=1)

            # OCR to extract name with custom config
            text = pytesseract.image_to_string(
                name_thresh, 
                config="--psm 7 -c tessedit_char_whitelist='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz -'"
            )
            
            name = clean_name(text)

            # Skip if no name could be extracted
            if not name:
                continue

            # Determine attendance score
            if len(faces) > 0:
                score = 1.0  # Present with video
            else:
                # Check if there's a muted mic icon (common in screenshots)
                # Look in the bottom left corner for a mic icon
                mic_area = gray[int(grid_h * 0.85):, :int(grid_w * 0.15)]
                _, mic_thresh = cv2.threshold(mic_area, 200, 255, cv2.THRESH_BINARY_INV)
                
                # If there's significant content in the mic area, assume participant is present without video
                if cv2.countNonZero(mic_thresh) > 50:
                    score = 0.5  # Present without video
                else:
                    score = 0.0  # Absent

            attendance[name] = score

    return attendance

if uploaded_file:
    image = Image.open(uploaded_file)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    st.image(image, caption="Uploaded Screenshot", use_column_width=True)

    rows = st.number_input("Number of Rows", min_value=1, max_value=10, value=4)
    cols = st.number_input("Number of Columns", min_value=1, max_value=10, value=9)

    if st.button("Process Attendance"):
        with st.spinner("Detecting attendance and extracting names..."):
            attendance = detect_attendance(image_cv, int(rows), int(cols))

        st.subheader("📊 Attendance Report")
        
        # Sort attendance by score (present with video first)
        sorted_attendance = sorted(attendance.items(), key=lambda x: x[1], reverse=True)
        
        # Display in a more organized way
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Present with video (1.0 point)**")
            for name, score in sorted_attendance:
                if score == 1.0:
                    st.write(f"✅ {name}")
        
        with col2:
            st.markdown("**Present without video (0.5 point)**")
            for name, score in sorted_attendance:
                if score == 0.5:
                    st.write(f"🟡 {name}")
        
        st.markdown("**Absent (0.0 point)**")
        absent_list = [name for name, score in sorted_attendance if score == 0.0]
        if absent_list:
            for name in absent_list:
                st.write(f"❌ {name}")
        else:
            st.write("No absent students detected")

        # Create and download CSV
        df = pd.DataFrame(sorted_attendance, columns=["Name", "Attendance Score"])
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="attendance_report.csv",
            mime="text/csv"
        )