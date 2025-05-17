import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(r"./input.mp4")

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
count = 0
tracker = Tracker()

# Define the line (x1, y1) to (x2, y2)
line = [(128, 332), (1005, 257)]
crossed_in = set()  # IDs of people who crossed into the area
crossed_out = set()  # IDs of people who crossed out of the area

def point_position(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

def is_crossing_line(prev_pos, curr_pos, line):
    (a, b) = line
    prev_side = point_position(a, b, prev_pos)
    curr_side = point_position(a, b, curr_pos)
    return prev_side * curr_side < 0

def crossing_direction(prev_pos, curr_pos, line):
    (a, b) = line
    prev_side = point_position(a, b, prev_pos)
    curr_side = point_position(a, b, curr_pos)
    if prev_side < 0 and curr_side >= 0:
        return "in"
    elif prev_side >= 0 and curr_side < 0:
        return "out"
    return None

prev_positions = {}

# Prepare video writer for output.mp4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1020, 500))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 2 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data

    px = pd.DataFrame(a).astype("float")
    bbox_list = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if c == 'person':
            bbox_list.append([x1, y1, x2, y2])

    bbox_idx = tracker.update(bbox_list)
    for bbox in bbox_idx:
        x3, y3, x4, y4, id = bbox
        cx = int((x3 + x4) / 2)
        cy = int((y3 + y4) / 2)

        prev_pos = prev_positions.get(id, (cx, cy))

        if is_crossing_line(prev_pos, (cx, cy), line):
            direction = crossing_direction(prev_pos, (cx, cy), line)
            if direction == "in":
                crossed_in.add(id)
            elif direction == "out":
                crossed_out.add(id)

        prev_positions[id] = (cx, cy)

        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), -1)
        cv2.putText(frame, str(int(id)), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

    cv2.line(frame, line[0], line[1], (0, 0, 255), 2)

    cv2.putText(frame, f"In: {len(crossed_in)}", (50, 80), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
    cv2.putText(frame, f"Out: {len(crossed_out)}", (50, 150), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)

    # Write the frame to output video instead of showing it
    out.write(frame)

cap.release()
out.release()
print("Processing complete. Output saved to output.mp4")
