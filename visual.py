from picamera2 import Picamera2
import cv2, time

picam2 = Picamera2()
picam2.start()
time.sleep(1)

for _ in range(100):
    frame = picam2.capture_array()
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cv2.destroyAllWindows()
picam2.stop()

picam2.start()
time.sleep(2)
picam2.capture_file("test.jpg")
picam2.stop()