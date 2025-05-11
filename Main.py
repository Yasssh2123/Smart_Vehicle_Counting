import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo11m.pt")
cap = cv2.VideoCapture(r"C:\Users\yashc\Downloads\Vehicle Detection project\Car_2Lane.mp4")

# Set the fixed output resolution to 1080x1920
output_width = 1920
output_height = 1080
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialize VideoWriter to save the output at the fixed resolution
out = cv2.VideoWriter("final_output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (output_width, output_height))

# Class name dictionary for vehicle types
class_names = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# Line drawing
drawing_points = []
def draw_lines(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_points.append((x, y))

ret, frame = cap.read()
cv2.namedWindow("Draw Lines")
cv2.setMouseCallback("Draw Lines", draw_lines)

print("Draw two points for UP line, then two for DOWN line. Press a key when done.")
while True:
    temp = frame.copy()
    if len(drawing_points) >= 2:
        cv2.line(temp, drawing_points[0], drawing_points[1], (0, 255, 255), 2)
    if len(drawing_points) >= 4:
        cv2.line(temp, drawing_points[2], drawing_points[3], (0, 0, 255), 2)
    cv2.imshow("Draw Lines", temp)
    if cv2.waitKey(1) != 255 and len(drawing_points) >= 4:
        break

cv2.destroyWindow("Draw Lines")
up_line = drawing_points[0:2]
down_line = drawing_points[2:4]

# Trackers
track_memory = {}
vehicle_id = 0
up_count = 0
down_count = 0
vehicle_classes = [2, 3, 5, 7]  # vehicle types

def intersect(p1, p2, q1, q2):
    def ccw(a,b,c): return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
    return ccw(p1,q1,q2) != ccw(p2,q1,q2) and ccw(p1,p2,q1) != ccw(p1,p2,q2)

def get_centroid(xyxy):
    x1, y1, x2, y2 = map(int, xyxy)
    return ((x1 + x2)//2, (y1 + y2)//2)

# Create a window with a fixed size (1080x1920)
cv2.namedWindow("Improved Vehicle Counter", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Improved Vehicle Counter", output_width, output_height)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = model.track(frame, conf=0.5, verbose=False)[0]
    new_memory = {}

    for box in detections.boxes:
        cls = int(box.cls[0])
        if cls not in vehicle_classes:
            continue

        class_name = class_names.get(cls, "Unknown")  # Get the class name from class ID
        xyxy = box.xyxy[0]
        cx, cy = get_centroid(xyxy)

        # Match with existing vehicles
        matched_id = None
        min_dist = 50
        for vid, data in track_memory.items():
            px, py = data["last_pos"]
            dist = np.hypot(cx - px, cy - py)
            if dist < min_dist:
                matched_id = vid
                min_dist = dist

        if matched_id is None:
            matched_id = vehicle_id
            vehicle_id += 1

        history = track_memory.get(matched_id, {"trace": [], "counted": False})
        trace = history["trace"] + [(cx, cy)]
        new_memory[matched_id] = {
            "trace": trace[-2:],  # Keep only last 2
            "counted": history["counted"],
            "last_pos": (cx, cy)
        }

        # Check crossing
        if not history["counted"] and len(trace) >= 2:
            p1, p2 = trace[-2], trace[-1]
            if intersect(p1, p2, up_line[0], up_line[1]):
                up_count += 1
                new_memory[matched_id]["counted"] = True
            elif intersect(p1, p2, down_line[0], down_line[1]):
                down_count += 1
                new_memory[matched_id]["counted"] = True

        # Draw box and display class name and ID
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, f"{class_name}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    track_memory = new_memory

    # Draw lines and counts
    cv2.line(frame, up_line[0], up_line[1], (0, 255, 255), 2)
    cv2.line(frame, down_line[0], down_line[1], (0, 0, 255), 2)
    cv2.putText(frame, f"UP: {up_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"DOWN: {down_count}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Resize frame to fit output resolution (1080x1920)
    resized_frame = cv2.resize(frame, (output_width, output_height))

    out.write(resized_frame)
    cv2.imshow("Improved Vehicle Counter", resized_frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
