from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# ----------------- CONFIGURATION -----------------
# 1. Path to your object detector (locates the seat)
# If you don't have 'last.pt', change this to 'yolov8n.pt'
OBJECT_MODEL_PATH = "yolo26x.pt" 

# 2. Path to your trained action classifier
ACTION_MODEL_PATH = "best_CLS.pt"

# 3. Video to analyze (Set to 0 for Webcam)
VIDEO_PATH = "1769078009331.mp4" 
# -------------------------------------------------

def main():
    # Verify models exist
    
    if not Path(ACTION_MODEL_PATH).exists():
        print(f"Error: Action model not found at {ACTION_MODEL_PATH}")
        return

    print("Loading models...")
    try:
        # Load the Object Detector (finds WHERE the action is)
        if Path(OBJECT_MODEL_PATH).exists():
            obj_model = YOLO(OBJECT_MODEL_PATH)
        else:
            print(f"Warning: {OBJECT_MODEL_PATH} not found. Downloading standard YOLOv8n...")
            obj_model = YOLO("yolov8n.pt") # Fallback to generic detector
            
        # Load the Action Classifier (finds WHAT is happening)
        action_model = YOLO(ACTION_MODEL_PATH)
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    video_url = "test_video.mp4"
    rtspurl = "rtsp://admin:500Net%4083504040@192.168.2.50:554/stream1"
    cap = cv2.VideoCapture(rtspurl)
    if not cap.isOpened():
        print(f"Error opening video: {VIDEO_PATH}")
        return
    print("Starting Inference. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. Detect Objects (Find the Seat)
        # using verbose=False to keep console clean
        results = obj_model(frame, verbose=False, conf=0.25)
        
        annotated_frame = frame.copy()
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 2. Extract the Region of Interest (ROI)
                # We crop the detected object + some margin context
                h, w = frame.shape[:2]
                margin = 30 # pixels context
                cx1 = max(0, x1 - margin)
                cy1 = max(0, y1 - margin)
                cx2 = min(w, x2 + margin)
                cy2 = min(h, y2 + margin)
                
                roi = frame[cy1:cy2, cx1:cx2]
                
                if roi.size == 0: continue

                # 3. Classify Action inside the ROI
                # The classifier looks at the cropped image and predicts the action
                cls_results = action_model(roi, verbose=False)
                
                if cls_results:
                    # Get top prediction
                    probs = cls_results[0].probs
                    top1_index = probs.top1
                    conf = probs.top1conf.item()
                    label = cls_results[0].names[top1_index]
                    
                    # Display label on screen
                    label_text = f"{label} ({conf:.0%})"
                    
                    # Color coding
                    color = (255, 255, 255) # White
                    if label == "stapling": color = (0, 0, 255)      # Red
                    if label == "fitting_cover": color = (0, 255, 0) # Green
                    
                    # Draw Box and Text
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, label_text, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Bike Action Inference", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()