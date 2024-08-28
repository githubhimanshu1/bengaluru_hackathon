import sys
import json
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from ultralytics import YOLO
from statsmodels.tsa.arima_model import ARIMA

def analyze_video(video_path, turning_patterns, model):
    counts_by_class_turning_pattern = {pattern: {
        'Bicycle': 0, 'Bus': 0, 'Car': 0, 'LCV': 0, 'Three Wheeler': 0, 'Truck': 0, 'Two Wheeler': 0
    } for pattern in turning_patterns}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return counts_by_class_turning_pattern

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for result in results:
            for obj in result:
                vehicle_class = obj.cls  # This should map to 'Bicycle', 'Bus', 'Car', etc.
                bbox = obj.xyxy  # Get bounding box coordinates
                centroid = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)  # Calculate centroid

                turning_pattern = determine_turning_pattern(centroid, turning_patterns)

                if turning_pattern and vehicle_class in counts_by_class_turning_pattern[turning_pattern]:
                    counts_by_class_turning_pattern[turning_pattern][vehicle_class] += 1

    cap.release()
    return counts_by_class_turning_pattern

def determine_turning_pattern(centroid, turning_patterns):
    # Define regions based on your camera views
    regions = {
        'A': ((100, 200), (300, 400)),  # Example region A (top-left and bottom-right coordinates)
        'B': ((400, 200), (600, 400)),  # Example region B
        # Add more regions as needed
    }

    entry_region = None
    exit_region = None

    for region, (top_left, bottom_right) in regions.items():
        if top_left[0] <= centroid[0] <= bottom_right[0] and top_left[1] <= centroid[1] <= bottom_right[1]:
            if not entry_region:
                entry_region = region
            else:
                exit_region = region
                break

    if entry_region and exit_region:
        pattern = entry_region + exit_region
        if pattern in turning_patterns:
            return pattern

    return None

def predict_future_counts(cumulative_counts):
    predicted_counts = {}

    for pattern, counts in cumulative_counts.items():
        predicted_counts[pattern] = {}
        for vehicle_class, count in counts.items():
            # Convert the counts to a time series for forecasting
            time_series = pd.Series([count] * 10)  # Example: use the same count repeated as a dummy time series

            # Fit ARIMA model (order should be tuned according to your data)
            model = ARIMA(time_series, order=(1, 1, 1))  # ARIMA(p, d, q) order parameters
            model_fit = model.fit(disp=0)
            
            # Forecast future values
            forecast = model_fit.forecast(steps=30)  # Forecast 30 time steps into the future
            predicted_count = forecast[0][-1]  # Get the last forecasted value
            
            predicted_counts[pattern][vehicle_class] = int(predicted_count)

    return predicted_counts

def main(input_json_path, output_json_path):
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    cam_id = list(data.keys())[0]
    vid_1_path = data[cam_id]['Vid_1']
    vid_2_path = data[cam_id]['Vid_2']

    # Load YOLO model (adjust path if needed)
    model = YOLO('yolov8n.pt')

    # Camera-specific turning patterns (this would come from the "Camera views and turning regions" file)
    turning_patterns = ['BC', 'BE', 'BG', 'DA', 'DE', 'DG', 'FA', 'FC', 'FG', 'HA', 'HC', 'HE']  # Example patterns

    # Analyze both video segments
    cumulative_counts_1 = analyze_video(vid_1_path, turning_patterns, model)
    cumulative_counts_2 = analyze_video(vid_2_path, turning_patterns, model)

    # Combine counts from both videos
    cumulative_counts = {pattern: {
        'Bicycle': cumulative_counts_1[pattern]['Bicycle'] + cumulative_counts_2[pattern]['Bicycle'],
        'Bus': cumulative_counts_1[pattern]['Bus'] + cumulative_counts_2[pattern]['Bus'],
        'Car': cumulative_counts_1[pattern]['Car'] + cumulative_counts_2[pattern]['Car'],
        'LCV': cumulative_counts_1[pattern]['LCV'] + cumulative_counts_2[pattern]['LCV'],
        'Three Wheeler': cumulative_counts_1[pattern]['Three Wheeler'] + cumulative_counts_2[pattern]['Three Wheeler'],
        'Truck': cumulative_counts_1[pattern]['Truck'] + cumulative_counts_2[pattern]['Truck'],
        'Two Wheeler': cumulative_counts_1[pattern]['Two Wheeler'] + cumulative_counts_2[pattern]['Two Wheeler']
    } for pattern in turning_patterns}

    # Predict future vehicle counts
    predicted_counts = predict_future_counts(cumulative_counts)

    # Prepare the output JSON
    output_data = {
        cam_id: {
            'Cumulative Counts': cumulative_counts,
            'Predicted Counts': predicted_counts
        }
    }

    # Write output to JSON file
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python app.py <input_json> <output_json>")
        sys.exit(1)

    input_json_path = sys.argv[1]
    output_json_path = sys.argv[2]

    main(input_json_path, output_json_path)
