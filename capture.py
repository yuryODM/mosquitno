import os
from picamera2 import Picamera2
import cv2
import time

picam2 = Picamera2()
picam2.start()
time.sleep(1)  # Warm-up

save_dir = "/home/mosquitno/Desktop/mosquitno/captured_images/"

num_images = 12
taken = 0

while taken < num_images:
    frame = picam2.capture_array()
    cv2.imshow("Preview", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

    if key == 32:  # SPACE key to take picture
        filename = os.path.join(save_dir, f"mosquito_10_{taken+1}.jpg")
        picam2.capture_file(filename)
        print(f"Saved {filename}")
        taken += 1
        time.sleep(0.3)  # small delay to avoid double-capture

cv2.destroyAllWindows()
picam2.stop()
