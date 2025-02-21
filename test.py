import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *

# Load YOLOv8 model
model = YOLO('yolov8s.pt')

# Load class names from coco.txt
with open("coco.txt", "r") as f:
    class_list = f.read().split("\n")

# Define tracking instance
tracker = Tracker()

# Define the polygon area for counting
area1 = [(170,261),(188,294),(321,262), (301, 224)]
area2=[(270,102),(286,134),(437,110),(415,70)]
# Mouse callback function
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print([x, y])

cv2.namedWindow('CV')
cv2.setMouseCallback('CV', RGB)

# Open video capture
cap = cv2.VideoCapture('peoplecount2.mp4')

frame_count = 0
people_count = 0
people_count2=0
tracked_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 3 != 0:  # Process every 3rd frame for efficiency
        continue

    frame = cv2.resize(frame, (1020, 500))

    # Perform YOLO detection
    results = model.predict(frame)
    detections = results[0].boxes.data

    # Convert to DataFrame
    px = pd.DataFrame(detections).astype("float")

    person_boxes = []

    # Process detected objects
    for _, row in px.iterrows():
        x1, y1, x2, y2, _, class_id = map(int, row[:6])

        # Only consider "person" class (class ID = 0 in COCO dataset)
        if class_id == 0:
            person_boxes.append([x1, y1, x2, y2])

    # Update tracker
    tracked_objects = tracker.update(person_boxes)

    for obj in tracked_objects:
        x3, y3, x4, y4, obj_id = obj
        cx, cy = (x3 + x4) // 2, (y3 + y4) // 2  # Compute centroid

        # Check if the centroid is inside the defined area
        if cv2.pointPolygonTest(np.array(area1, np.int32), (cx, cy), False) >= 0:
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
            cv2.circle(frame, (cx,cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f'ID: {obj_id}', (x3, y3 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Count unique objects
            if obj_id not in tracked_ids:
                tracked_ids.add(obj_id)
                people_count += 1
        
        if cv2.pointPolygonTest(np.array(area2, np.int32), (cx, cy), False) >= 0:
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
            cv2.circle(frame, (cx,cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f'ID: {obj_id}', (x3, y3 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            if obj_id not in tracked_ids:
                tracked_ids.add(obj_id)
                people_count2 += 1

    # Draw polygon area
    cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 0, 255), 2)
    
    cv2.polylines(frame,[np.array(area2,np.int32)],True,(0,0,255),2)

    # Display count
    cv2.putText(frame, f'People Went Up: {people_count}', (737, 432), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(frame,f'People Went Down: {people_count2}',(738,479),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)
    cv2.imshow("CV", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
