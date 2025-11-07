import time
import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

# Initialize camera
picam2 = Picamera2()
picam2.start()
time.sleep(1)

# Load YOLO model
model = YOLO('/home/mosquitno/Desktop/mosquitno/mosquito_train/exp15/weights/best.pt')  # replace with your trained weights

try:
    while True:
        frame = picam2.capture_array()

        # Convert 4-channel frames to 3-channel BGR
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Run YOLO detection
        results = model(frame)

        # Draw bounding boxes
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            for box, score in zip(boxes, scores):
                if score < 0.4:
                    continue
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{score:.2f}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Mosquito Tracking", frame)
        if cv2.waitKey(1) == 27:  # ESC to exit
            break

finally:
    cv2.destroyAllWindows()
    picam2.stop()
