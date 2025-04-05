import cv2
from ultralytics import YOLO
import os

model = YOLO("best.pt")
try:
    model.fuse()
except AttributeError:
    print("Warning: Model fusion not supported. Proceeding without fusion.")


video_path = "sn.mp4"
if not os.path.exists(video_path):
    print(f"Error: Video file '{video_path}' not found.")
    exit()

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error or Processing is completed")
        break
    results = model.predict(frame, conf=0.5, show=True, stream=True, save=True)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = box.cls[0]
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("YOLOv8 Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
