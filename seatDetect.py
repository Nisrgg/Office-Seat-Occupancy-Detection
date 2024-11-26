import cv2
import torch
import time
import signal
import sys

# Load YOLOv5s model for a balance of speed and accuracy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'yolov5m').to(device)

# Set confidence threshold and allowed classes
model.conf = 0.5  # Confidence threshold
model.iou = 0.45  # IoU threshold for non-max suppression
ALLOWED_CLASSES = ['person', 'chair']

# Path to your uploaded video
video_path = 'vid2.mp4'
cap = cv2.VideoCapture(video_path)

# Initialize data structures to track chair states and times
chair_states = {}  # Tracks if a chair is occupied {chair_id: True/False}
chair_times = {}   # Stores the total occupied time for each chair {chair_id: total_time}
chair_start_times = {}  # Tracks the start time of occupancy {chair_id: start_time}

# Function to handle program exit and print total chair usage times
def handle_exit(signal, frame):
    print("\nProgram interrupted or exiting. Printing chair occupancy times:")
    for chair_id, total_time in chair_times.items():
        print(f"{chair_id}: {total_time:.2f} seconds")
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

# Set up signal handler to catch exit or interrupt
signal.signal(signal.SIGINT, handle_exit)  # Handle Ctrl+C interruption
signal.signal(signal.SIGTERM, handle_exit)  # Handle termination

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to capture video. Exiting...")
        break

    # Object detection
    start_time = time.time()  # Start timer for inference
    results = model(frame)
    detections = results.xyxyn[0]
    inference_time = time.time() - start_time

    people = []
    chairs = []

    # Process each detection
    for det in detections:
        confidence, label_idx = det[4], int(det[5])
        label = model.names[label_idx]

        if confidence > model.conf and label in ALLOWED_CLASSES:
            x1, y1, x2, y2 = det[:4]
            x1, y1, x2, y2 = int(x1 * frame.shape[1]), int(y1 * frame.shape[0]), int(x2 * frame.shape[1]), int(y2 * frame.shape[0])

            # Classify detections
            if label == 'person':
                people.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)

            elif label == 'chair':
                # Filter based on bounding box size
                box_area = (x2 - x1) * (y2 - y1)
                frame_area = frame.shape[0] * frame.shape[1]
                if 0.01 * frame_area < box_area < 0.1 * frame_area:  # Ignore too small/large objects
                    chair_id = f"chair_{len(chairs)}"
                    chairs.append((x1, y1, x2, y2, chair_id))

                    # Initialize chair tracking data if not already present
                    if chair_id not in chair_states:
                        chair_states[chair_id] = False
                        chair_times[chair_id] = 0
                        chair_start_times[chair_id] = 0

                    # Draw bounding box for the chair
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"Chair {len(chairs)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Check for overlap between chairs and people
    for cx1, cy1, cx2, cy2, chair_id in chairs:
        person_detected_near_chair = False

        for px1, py1, px2, py2 in people:
            # Check for bounding box overlap
            if px1 < cx2 and px2 > cx1 and py1 < cy2 and py2 > cy1:
                person_detected_near_chair = True
                break

        # Update chair occupancy status and timing
        if person_detected_near_chair:
            if not chair_states[chair_id]:  # Chair becomes occupied
                chair_states[chair_id] = True
                chair_start_times[chair_id] = time.time()  # Start occupancy timer
        else:
            if chair_states[chair_id]:  # Chair becomes empty
                chair_states[chair_id] = False
                # Update total occupied time
                chair_times[chair_id] += time.time() - chair_start_times[chair_id]
                chair_start_times[chair_id] = 0  # Reset start time

        # Display occupancy status for each chair
        status = "Occupied" if person_detected_near_chair else "Empty"
        cv2.putText(frame, f"Status: {status}", (cx1, cy2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Debugging print statements
    for chair_id, occupied in chair_states.items():
        print(f"Chair {chair_id} occupied: {occupied}, start time: {chair_start_times.get(chair_id, 0)}")



    # Show the frame
    cv2.imshow('Seat Occupancy Detection', frame)

    # Break the loop on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Print total occupied times for each chair
print("\nTotal Occupied Times for Each Chair:")
for chair_id, total_time in chair_times.items():
    print(f"{chair_id}: {total_time:.2f} seconds")