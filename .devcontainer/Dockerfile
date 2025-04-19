FROM mcr.microsoft.com/devcontainers/python:3.10

# Install system dependencies
RUN apt-get update && apt-get install -y libgl1 tesseract-ocr

# Set Tesseract path
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
