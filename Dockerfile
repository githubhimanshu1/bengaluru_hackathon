# Use the official Python base image
FROM python:3.8-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update \
    && apt-get install -y \
        curl \
        build-essential \
        libssl-dev \
        libffi-dev \
        libatlas-base-dev \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Download YOLOv8 model weights using curl
RUN curl -L -o yolov8n.pt https://github.com/ultralytics/yolov5/releases/download/v8.0/yolov8n.pt

# Define the command to run the application
CMD ["python", "app.py", "input_file.json", "output_file.json"]
