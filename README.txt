This is a basic sample code to Detect an object i.e. Person in real-time.

To run the code:
    1. Install packages - opencv, torch, ultralytics, YOLO.
    2. Set the cv2.VideoCapture() to appropriate index, 0 for default, 1 or 2 for external camera.
    3. Run the code.
    4. press "q" to exit the camera window and terminate.

This code uses pre-trained YOLOv3 to target the object- "Person" and detects only persons and nothing else.