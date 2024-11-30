import cv2
from ultralytics import YOLO
import os
import pyttsx3

# Verify YOLO model path
model_path = r"C:\Users\ASUS\OneDrive\Desktop\SDP\best.pt"
if not os.path.exists(model_path):
    print("Error: YOLO model file not found.")
    exit()

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

# Load YOLO model
try:
    model = YOLO(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load YOLO model. Error: {e}")
    exit()

# Check the class names loaded by the model
print("Class names in the model:")
print(model.names)  # This will print out all the class names the model can detect

# Process each frame and run YOLO detection
def process_frame(frame):
    try:
        # Convert frame to RGB for YOLO
        img_rgb = cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_BGR2RGB)

        # Run YOLO detection
        results = model.predict(img_rgb, conf=0.3, save=False)  # Adjusted confidence threshold to 0.3

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[int(box.cls[0])]  # Get class label
                conf = float(box.conf[0])  # Get confidence score

                # Debugging: Print all detected labels and confidence
                print(f"Detected: {label}, Confidence: {conf}")

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label with background
                label_text = f"{label} {conf:.2f}"
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 5), 
                              (x1 + label_size[0] + 10, y1), (0, 255, 0), -1)
                cv2.putText(frame, label_text, (x1 + 5, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                # Trigger voice output for specific labels
                if label.lower() in {"no parking", "stop sign"}:
                    engine.say(f"{label} detected")
                    engine.runAndWait()
    except Exception as e:
        print(f"Error processing frame: {e}")
    return frame

# Capture video from webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Press 'q' to quit.")
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Process and display frame
        cv2.imshow('YOLO Object Detection', process_frame(frame))

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")
