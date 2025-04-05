# Project Overview

This project implements a snooker score counting system using the YOLOv8 object detection model. The system processes a video of a snooker game, detects balls, and calculates the score dynamically based on the color of the balls that cross a user-defined line. The program also saves the processed video with bounding boxes, the score, and the detected ball colors.

## Features

### Dynamic Line Drawing
- The user can draw a line on the video by clicking two points on the frame.
- The program waits for the user to define the line before starting the detection and scoring process.

### Object Detection
- YOLOv8 is used to detect snooker balls in the video.
- Bounding boxes and labels are drawn around detected objects.

### Score Calculation
- The score is calculated based on the color of the ball that crosses the user-defined line:
  - Red: 1 point
  - Yellow: 2 points
  - Green: 3 points
  - Brown: 4 points
  - Blue: 5 points
  - Pink: 6 points
  - Black: 7 points
- The color of the ball and the updated score are displayed on the video.

### Video Output
- The processed video is saved as `output.mp4` with bounding boxes, the user-defined line, and the score.

### Real-Time Display
- The video is displayed in real-time with all the visual elements (line, bounding boxes, score, and ball color).

## Requirements

- Python 3.8 or higher
- Required Python libraries:
  - `opencv-python`
  - `ultralytics`
  - `os` (built-in)

## Setup Instructions

### Clone the Repository
[Add the cloning command here if applicable, e.g., `git clone <repository-url>`]

### Install Dependencies
Install the required Python libraries using pip:

```bash
pip install opencv-python ultralytics
```

### **Prepare the YOLOv8 Model**
- Place the YOLOv8 model file (best.pt) in the project directory.
- Ensure the model is trained to detect snooker ball colors (e.g., red, yellow, green, etc.).

### Add the Video File
- Place the video file (sn.mp4 or whatever name) in the project directory.

## How to Run
### Run the script:
```bash
python count.py
```

### Follow the on-screen instructions:
- Click two points on the video frame to draw a line.
- Once the line is drawn, the program will start detecting balls and calculating the score.
- Press q to quit the program at any time.
- The processed video will be saved as output.mp4 in the project directory.
