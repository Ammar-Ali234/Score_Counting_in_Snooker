import cv2
from ultralytics import YOLO
import os

# Initialize global variables for line drawing
line_points = []
drawing_line = False
line_drawn = False

# Snooker ball scores based on color
snooker_scores = {
    "red": 1,
    "yellow": 2,
    "green": 3,
    "brown": 4,
    "blue": 5,
    "pink": 6,
    "black": 7
}

# Mouse callback function to capture line points
def draw_line(event, x, y, flags, param):
    global line_points, drawing_line, line_drawn
    if event == cv2.EVENT_LBUTTONDOWN and not line_drawn:
        if len(line_points) < 2:
            line_points.append((x, y))
        if len(line_points) == 2:
            drawing_line = True
            line_drawn = True
            print(f"Line drawn between points: {line_points[0]} and {line_points[1]}")

# Load YOLO model
model = YOLO("best.pt")
try:
    model.fuse()
except AttributeError:
    print("Warning: Model fusion not supported. Proceeding without fusion.")

# Check if video file exists
video_path = "sn.mp4"
if not os.path.exists(video_path):
    print(f"Error: Video file '{video_path}' not found.")
    exit()

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Get video properties for saving the output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for saving the video
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

# Set up the mouse callback
cv2.namedWindow("YOLOv8 Detection")
cv2.setMouseCallback("YOLOv8 Detection", draw_line)

# Initialize score
score = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error or Processing is completed")
        break

    # Wait for the user to draw the line
    if not line_drawn:
        cv2.putText(frame, "Click two points to draw the line", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if len(line_points) == 1:
            cv2.circle(frame, line_points[0], 5, (0, 255, 0), -1)
        cv2.imshow("YOLOv8 Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Draw the line if two points are selected
    if len(line_points) == 2:
        cv2.line(frame, line_points[0], line_points[1], (0, 0, 255), 2)

    # Perform object detection
    results = model.predict(frame, conf=0.5, show=False, stream=True, save=False)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = box.cls[0]
            label = f"{model.names[int(cls)]} {conf:.2f}"

            # Draw bounding box and label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (240, 240, 57), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 57), 2)

            # Check if the object crosses the line
            if len(line_points) == 2:
                # Calculate the center of the bounding box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # Check if the center crosses the line
                x1_line, y1_line = line_points[0]
                x2_line, y2_line = line_points[1]
                if min(y1_line, y2_line) <= center_y <= max(y1_line, y2_line) and min(x1_line, x2_line) <= center_x <= max(x1_line, x2_line):
                    # Get the color of the ball and update the score
                    ball_color = model.names[int(cls)]
                    ball_score = snooker_scores.get(ball_color, 0)
                    score += ball_score
                    print(f"Ball potted: {ball_color}, Score: {score}")

                    # Display the ball color and score on the frame
                    cv2.putText(frame, f"Potted: {ball_color}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the score on the frame
    cv2.putText(frame, f"Score: {score}", (10, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)

    # Write the frame to the output video
    out.write(frame)

    # Show the frame
    cv2.imshow("YOLOv8 Detection", frame)
    if cv2.waitKey(150) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()