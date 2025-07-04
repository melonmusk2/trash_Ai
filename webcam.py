import cv2
import datetime
import os
import numpy as np 
from urllib.request import urlopen
from PIL import Image
import timm
import torch
import subprocess
import time
from Data import train
from rembg import remove
from PIL import Image
import io


# --- Configuration ---
WEBCAM_INDEX = 0  # Usually 0 for the default webcam
OUTPUT_DIR = 'motion_detected_images' # Directory to save images
MIN_AREA = 5000  # Minimum contour area to be considered a significant object (adjust this value!)
                  # Smaller values detect smaller movements, larger values ignore small noise.
                  # needs to be adjusted according to env and webcam noise behavoir
CAPTURE_COOLDOWN_SECONDS = 3 # Time in seconds before another picture can be taken AFTER a capture    
MOTION_DURATION_REQUIRED = 2              
GAUSSIAN_BLUR_SIZE = (21, 21) # Blur applied to frames to reduce noise
THRESHOLD_DELTA = 25 # Pixel intensity difference threshold for motion detection

# Create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Open the webcam
cap = cv2.VideoCapture(WEBCAM_INDEX)

if not cap.isOpened():
    print(f"Error: Could not open webcam with index {WEBCAM_INDEX}")
    exit()

 # Expanded category mapping

print("Webcam opened successfully. Press 'q' to quit.")
print(f"Waiting to establish background. Please keep the scene clear for a few seconds...")

# Initialize background frame
avg_background_frame = None # This will store the accumulated background as float
background_ready = False    # Flag to indicate if background is established
frame_count = 0
BACKGROUND_ESTIMATION_FRAMES = 10

last_capture_time = None
motion_start_time = None # Tracks when sustained motion began
category = None
return_code = 1

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_SIZE, 0)
    current_time = datetime.datetime.now() # Get current time for all time-based checks

    # --- Background Establishment Phase ---
    if not background_ready:
        if avg_background_frame is None:
            avg_background_frame = gray.astype("float")
        else:
            cv2.accumulateWeighted(gray, avg_background_frame, 0.5)

        frame_count += 1

        if frame_count >= BACKGROUND_ESTIMATION_FRAMES:
            background_ready = True
            print("Background established. Starting motion detection...")
            avg_background_frame_uint8 = cv2.convertScaleAbs(avg_background_frame)
        

        continue # Skip the motion detection logic until background is ready

    # --- Motion Detection Phase ---
    frame_delta = cv2.absdiff(avg_background_frame_uint8, gray)
    thresh = cv2.threshold(frame_delta, THRESHOLD_DELTA, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    object_detected_in_current_frame = False
    for contour in contours:
        if cv2.contourArea(contour) < MIN_AREA:
            continue

        object_detected_in_current_frame = True
        (x, y, w, h) = cv2.boundingRect(contour)

    # --- Sustained Motion Logic ---
    if object_detected_in_current_frame:
        if motion_start_time is None:
            # Motion just started
            motion_start_time = current_time
        # else: motion is continuing, motion_start_time is already set
    else:
        # No motion detected in this frame, reset motion_start_time
        motion_start_time = None

    # --- Image Capture Logic ---
    capture_condition_met = False
    num_motion_files = len([name for name in os.listdir(OUTPUT_DIR) if os.path.isfile(os.path.join(OUTPUT_DIR, name))])

    if motion_start_time is not None: # Motion is currently ongoing
        time_since_motion_start = (current_time - motion_start_time).total_seconds()
        if time_since_motion_start >= MOTION_DURATION_REQUIRED:
            # Motion has been sustained for long enough
            if last_capture_time is None or \
               (current_time - last_capture_time).total_seconds() > CAPTURE_COOLDOWN_SECONDS:
                capture_condition_met = True
                if num_motion_files < 1: # This means 0 files
                  capture_condition_met = True

    if capture_condition_met and num_motion_files == 0 and return_code is not None:
        filename = os.path.join(OUTPUT_DIR, f"object_motion.png")
        print("Motion detected!")
        cv2.imwrite(filename, frame)
        last_capture_time = current_time
        # Important: Reset motion_start_time AFTER capturing
        # This prevents taking multiple pictures within the same sustained motion event
        # until the cooldown or new motion starts.
        motion_start_time = None 

    if num_motion_files > 0:
        try:
          input_path = "motion_detected_images/object_motion.png"
          output_path = "motion_detected_images/object_motion.png"
          output_path2 = "obj.png"

          with open(input_path, "rb") as f:
            input_image = f.read()

          img1 = remove(input_image)

          
          with open(output_path, "wb") as out:
           out.write(img1)
          
          img = Image.open('motion_detected_images/object_motion.png').convert("RGB")
        except Exception as e:
            print(f"An error occurred while loading the image: {e}")
            exit()

        label = train(img)
        # Display result
        if label:
            category = label
            os.remove("motion_detected_images/object_motion.png")
        else:
            print("Could not display human-readable label.")
            os.remove("motion_detected_images/object_motion.png")
    if category is not None:
     arg = {
      "organic": "1",
      "anything else": "2",
      "Paper": "3",
      "plastik": "4",
      "batteries": "5",
      "glass": "5"
     }

     ARG = arg.get(category)
     category = None

     command = [
      "ssh",
      "robot@ev3dev",
      "brickrun",
      "--directory=/home/robot/motor_control_to_sort_trash",
      "/home/robot/motor_control_to_sort_trash/main.py",
      ARG
     ]
    
     process = subprocess.Popen(command)

     time.sleep(6)

     return_code = process.poll()

     if return_code is None:
        print("The subprocess is still running after 6000ms delay.")
    # You can choose to terminate it, or wait for it to finish
    # For example, to wait for it to finish:
    # process.wait()
    # print(f"Subprocess finished with exit code: {process.returncode}")
     else:
        print(f"The subprocess has finished after 2000ms delay with exit code: {return_code}") 

