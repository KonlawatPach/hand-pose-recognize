from picamera2 import Picamera2, Preview
from time import sleep

picam2 = Picamera2()
picam2.start_and_capture_file("now_image.jpg")
picam2.close()