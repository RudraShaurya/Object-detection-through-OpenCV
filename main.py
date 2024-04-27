#ultralytics needs to be installed
import cv2
import torch
from PIL import Image
from ultralytics import YOLO

# Load the pre-trained YOLOv3 object detection model
model = YOLO("yolov3.pt")

# Set the object class you want to detect (e.g.- 'person')
target_class = 'person'

# Initialize the video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, 1 or 2 for external cameras, video file path as well

while True:
    # Capture a frame from the video
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("Failed to read frame from the video. Exiting.")
        break

    # Preprocess the frame for the object detection model
    img = Image.fromarray(frame)
    results = model(img, stream=True)

    # Initialize a list to store the bounding box coordinates
    bounding_boxes = []

    # Parse the output and draw the bounding boxes
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_id = int(box.cls[0])
            if target_class == model.names[class_id]:
                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add the bounding box coordinates to the list
                bounding_boxes.append((x1, y1, x2 - x1, y2 - y1))

    # Display the resulting frame
    cv2.imshow('Object Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

# Print the list of bounding box coordinates
print(bounding_boxes)