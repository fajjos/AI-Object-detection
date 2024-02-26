import cv2
import numpy as np
import webbrowser

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
clicked_object_index = None

# Function to handle mouse click event
def mouse_callback(event, x, y, flags, param):
    global open_search, clicked_object_index
    if event == cv2.EVENT_LBUTTONDOWN:
        open_search = True
        clicked_object_index = None
        for i, (box, label) in enumerate(zip(boxes, labels)):
            if box[0] <= x <= box[0] + box[2] and box[1] <= y <= box[1] + box[3]:
                clicked_object_index = i
                break

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

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes on the frame
    labels = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            labels.append(label)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the augmented frame
    cv2.imshow('Object Detection', frame)

    # Check if search flag is set and open the search tab for the clicked object
    if open_search and clicked_object_index is not None:
        clicked_label = str(labels[clicked_object_index])
        print(f"Clicked on: {clicked_label}")
        search_url = f"https://www.google.com/search?q={clicked_label}"
        webbrowser.open(search_url)
        open_search = False

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
