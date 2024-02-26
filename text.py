import cv2
import numpy as np
import pytesseract
import webbrowser

# Set the path to the Tesseract executable (change it based on your installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []

with open("C:\\Users\\FAJJOS\\OneDrive\\Desktop\\AR object detection\\coco.names.txt", "r") as f:
    classes = [line.strip() for line in f]

layer_names = net.getUnconnectedOutLayersNames()

# Initialize the video capture object (0 for the default camera, or provide the camera device index)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Create a flag to track whether to open the search tab
open_search = False
clicked_coordinates = None
detect_objects = True
detect_text = True

# Lists to store detected objects and text
detected_objects = []
detected_text = []

# Function to handle mouse click event
def mouse_callback(event, x, y, flags, param):
    global open_search, clicked_coordinates, detected_objects
    if event == cv2.EVENT_LBUTTONDOWN:
        open_search = True
        clicked_coordinates = (x, y)

        # Check if object detection is enabled
        if detect_objects:
            min_distance = float('inf')
            closest_object_index = None

            for i, (box, label) in enumerate(detected_objects):
                box_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                distance = np.linalg.norm(np.array(clicked_coordinates) - np.array(box_center))

                if distance < min_distance:
                    min_distance = distance
                    closest_object_index = i

            if closest_object_index is not None:
                clicked_label = str(detected_objects[closest_object_index][1])
                print(f"Clicked on: {clicked_label}")

                # Check if search is enabled
                if open_search:
                    search_url = f"https://www.google.com/search?q={clicked_label}"
                    webbrowser.open(search_url)

        open_search = False
        clicked_coordinates = None

# Set the mouse callback function
cv2.namedWindow("Object Detection")
cv2.setMouseCallback("Object Detection", mouse_callback)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Check if the frame was successfully captured
    if not ret:
        print("Error: Could not read frame.")
        break

    height, width, channels = frame.shape

    # Detect objects
    if detect_objects:
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(layer_names)

        # Get class names and draw bounding boxes
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        # Draw bounding boxes on the frame
        detected_objects = []
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            detected_objects.append(([x, y, w, h], label))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Perform OCR on the frame if text detection is enabled
    if detect_text:
        try:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray_frame)
            detected_text = text.split('\n')
        except pytesseract.pytesseract.TesseractError as e:
            print(f"Tesseract Error: {e}")
            detected_text = []

    # Display the augmented frame
    cv2.imshow('Object Detection', frame)

    # Display detected objects and text in a sidebar
    sidebar_text = "Detected Objects:\n"
    for _, label in detected_objects:
        sidebar_text += f"- {label}\n"

    sidebar_text += "\nDetected Text:\n"
    for text_line in detected_text:
        sidebar_text += f"- {text_line}\n"

    cv2.putText(frame, sidebar_text, (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Check for keyboard input to toggle detection and search
    key = cv2.waitKey(1) & 0xFF
    if key == ord('o'):
        detect_objects = not detect_objects
    elif key == ord('t'):
        detect_text = not detect_text
    elif key == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
