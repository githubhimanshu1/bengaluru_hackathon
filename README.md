# Bengaluru Mobility Challenge Phase 1

## Overview

This project aims to address the Bengaluru Mobility Challenge Phase 1 by developing a Dockerized solution for vehicle counting and prediction from video data. The solution processes video clips from multiple cameras to estimate and predict vehicle counts using advanced machine learning techniques.

## Project Structure

- `app.py`: The main script for processing video clips and generating predictions.
- `requirements.txt`: Lists the Python libraries and versions required for the project.
- `Dockerfile`: Instructions to build the Docker image.
- `README.md`: This file, providing an overview and instructions.
- `team_name_report.pdf`: Detailed report including methodology, results, and conclusions.

## Docker Image

### Building the Docker Image

1. Ensure Docker is installed on your system.
2. Build the Docker image using the following command:

   ```sh
 docker build -t snow_rangers . 
