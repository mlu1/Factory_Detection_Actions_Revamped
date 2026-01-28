from collections import defaultdict, deque, Counter
import cv2
import numpy as np
from ultralytics import YOLO
import threading

class VideoStream:
    def read(self):
       
        with self.lock:
            return self.ret, self.frame

    def isOpened(self):
        return not self.stopped

    def release(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()
   
    def __init__(self, src, name="VideoStream"):
        LOW_LATENCY_BUFFER_SIZE = 1
        if isinstance(src, int):
            self.cap = cv2.VideoCapture(src)
        else:
            self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, LOW_LATENCY_BUFFER_SIZE)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 讀取第一幀以初始化狀態
        self.grabbed, self.frame = self.cap.read()
        self.ret = self.grabbed

        self.lock = threading.Lock()
        self.ready = threading.Event()
        if self.grabbed:
            self.ready.set()

        self.stopped = False
        self.thread = threading.Thread(target=self.update, name=name, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            try:
                ret = self.cap.grab()
                ret_retrieve, frame = self.cap.retrieve()
            except Exception:
                # 發生錯誤時停止線程
                self.stopped = True
                #顯示錯誤訊息
                print("Error grabbing/retrieving frame from VideoCapture.")       
                break

            with self.lock:
                self.ret = ret
                if ret_retrieve:
                    self.frame = frame

            self.ready.set()

        self.cap.release()

if __name__ == "__main__":
    # 1. Load Object Model (Your custom model)
    object_model = YOLO("yolo26x.pt") 

    # 2. Load Pose Model (For hand/wrist detection)
    # This will automatically download if you don't have it
    pose_model = YOLO("yolov8n-pose.pt")

    # Open the video file
    video_url = "test_video.mp4"
    rtspurl = "rtsp://admin:500Net%4083504040@192.168.2.50:554/stream1"
    #cap = VideoStream(rtspurl, name="RTSPStream")
    #cap = VideoCapture(video_url, cv2.CAP_FFMPEG)
    cap = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG)

    # Store the track history
    track_history = defaultdict(lambda: [])
    
    # Store interaction history to compare Hand vs Object movement
    # Key: track_id, Value: list of (obj_x, obj_y, wrist_x, wrist_y)
    interaction_history = defaultdict(lambda: [])
    
    # Store object interactions
    object_interactions = defaultdict(lambda: "None")

    # --- Smoothing State ---
    # Store recent raw actions for majority voting
    action_history = defaultdict(lambda: deque(maxlen=10)) 
    # Store the last confirmed action to prevent flickering
    confirmed_actions = defaultdict(lambda: "None")

    def is_point_in_box(point, box, margin=20):
        """Check if a point (x, y) is inside or close to a box (x, y, w, h)."""
        px, py = point
        bx, by, bw, bh = box
        x1 = bx - bw / 2 - margin
        y1 = by - bh / 2 - margin
        x2 = bx + bw / 2 + margin
        y2 = by + bh / 2 + margin
        return x1 <= px <= x2 and y1 <= py <= y2

    # Loop through the video frames
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # --- A. Object Detection & Tracking ---
            obj_results = object_model.track(frame, persist=True, verbose=False)[0]
            
            yolo_boxes = []
            track_ids = []
            if obj_results.boxes and obj_results.boxes.is_track:
                yolo_boxes = obj_results.boxes.xywh.cpu().numpy()
                track_ids = obj_results.boxes.id.int().cpu().tolist()
                
                # Draw tracks
                frame = obj_results.plot() # Plot objects
                
                for box, track_id in zip(yolo_boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y))) 
                    if len(track) > 30: track.pop(0)

                    # Draw trail
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=3)

                    # Draw Action Label
                    action = object_interactions[track_id]
                    if action != "None":
                        cv2.putText(frame, action, (int(x - w/2), int(y - h/2) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # --- B. Pose/Hand Detection (No MediaPipe) ---
            pose_results = pose_model(frame, verbose=False)[0]
            

            # Collect all wrist positions from all detected persons
            wrists = []
            if pose_results.keypoints is not None:
                # Keypoints shape: (Num_Persons, 17, 2 or 3)
                # COCO Keypoint 9 = Left Wrist, 10 = Right Wrist
                kpts = pose_results.keypoints.xy.cpu().numpy()
                for person in kpts:
                    if len(person) > 10:
                        # Check confidence if available, or just take coords
                        # Assuming (x,y) format. 
                        if person[9][0] > 0 and person[9][1] > 0: wrists.append(person[9]) # L Wrist
                        if person[10][0] > 0 and person[10][1] > 0: wrists.append(person[10]) # R Wrist
            
            # Visualize wrists (Optional debug)
            for wx, wy in wrists:
                cv2.circle(frame, (int(wx), int(wy)), 5, (0, 255, 255), -1)

            # --- C. Interaction Logic ---
            
            for box, track_id in zip(yolo_boxes, track_ids):
                # 1. Find the closest contacting wrist
                is_touching = False
                closest_wrist = None
                min_dist = float('inf')

                for wrist in wrists:
                    # Check if inside box with margin
                    if is_point_in_box(wrist, box, margin=15):
                        is_touching = True
                        # Euclidean distance to box center
                        dist = ((wrist[0] - box[0])**2 + (wrist[1] - box[1])**2)**0.5
                        if dist < min_dist:
                            min_dist = dist
                            closest_wrist = wrist
                
                # 2. Update Interaction History (track both object and wrist positions)
                int_hist = interaction_history[track_id]
                if is_touching and closest_wrist is not None:
                    # Append (obj_x, obj_y, wrist_x, wrist_y)
                    int_hist.append((box[0], box[1], closest_wrist[0], closest_wrist[1]))
                    if len(int_hist) > 10: int_hist.pop(0)  # Keep last 10 frames
                else:
                    # Reset history if contact broken
                    interaction_history[track_id] = []

                # 3. Determine Raw Action (Instantaneous) based on co-movement
                raw_action = "None"

                if is_touching and len(int_hist) >= 5:
                    # Calculate deltas over last 5 frames
                    curr = int_hist[-1]
                    prev = int_hist[-5]  # 5 frames ago (~0.15s at 30fps)
                    
                    obj_dy = curr[1] - prev[1]    # Object vertical movement
                    obj_dx = curr[0] - prev[0]    # Object horizontal movement
                    wrist_dy = curr[3] - prev[3]  # Wrist vertical movement
                    wrist_dx = curr[2] - prev[2]  # Wrist horizontal movement
                    
                    # Movement thresholds
                    UP_THRESH = -3.0
                    DOWN_THRESH = 3.0
                    MOVE_TOGETHER_TOL = 15  # Tolerance for "moving together"
                    
                    # Check "Moving Together" heuristic:
                    # Both moving up significantly
                    if obj_dy < UP_THRESH and wrist_dy < UP_THRESH:
                        diff = abs(obj_dy - wrist_dy)
                        if diff < MOVE_TOGETHER_TOL:
                            raw_action = "Picking Up"
                        else:
                            raw_action = "Handling"
                    
                    # Both moving down significantly
                    elif obj_dy > DOWN_THRESH and wrist_dy > DOWN_THRESH:
                        diff = abs(obj_dy - wrist_dy)
                        if diff < MOVE_TOGETHER_TOL:
                            raw_action = "Putting Down"
                        else:
                            raw_action = "Handling"
                    
                    # Object is stationary (Processing or Holding)
                    elif abs(obj_dy) < 2.0 and abs(obj_dx) < 2.0:
                        # If hand is moving while object is still -> Processing
                        hand_movement = (wrist_dx**2 + wrist_dy**2)**0.5
                        if hand_movement > 3.0:
                            raw_action = "Processing"
                        else:
                            raw_action = "Holding"
                    
                    # Horizontal movement (Carrying)
                    elif abs(obj_dx) > 3.0 and abs(wrist_dx) > 3.0:
                        diff = abs(obj_dx - wrist_dx)
                        if diff < MOVE_TOGETHER_TOL:
                            raw_action = "Carrying"
                        else:
                            raw_action = "Handling"
                    else:
                        raw_action = "Handling"  # Generic handling
                
                elif is_touching:
                    raw_action = "Touching"  # Not enough history yet

                # 4. Smoothing / Majority Voting to prevent flickering
                action_history[track_id].append(raw_action)
                
                # Get the most common action in the last 10 frames
                counts = Counter(action_history[track_id])
                if counts:
                    most_common, frequency = counts.most_common(1)[0]
                    
                    # Confidence threshold for state change
                    if most_common == "None":
                        # Release immediately if mostly None
                        confirmed_actions[track_id] = "None"
                    elif frequency >= 6:
                        # Require consistency (6/10 frames) to switch state
                        confirmed_actions[track_id] = most_common
                    # If not confident, keep the previous confirmed action
                    
                    object_interactions[track_id] = confirmed_actions[track_id]

            cv2.imshow("YOLO HOI (Pose)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()