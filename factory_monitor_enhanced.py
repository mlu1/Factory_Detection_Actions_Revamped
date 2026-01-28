from collections import defaultdict, deque, Counter
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import json
import datetime
import sqlite3
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
import time

@dataclass
class EmployeeSession:
    """Track an employee's work session"""
    employee_id: str
    start_time: datetime.datetime
    current_activity: str = "None"
    items_completed: int = 0
    total_handling_time: float = 0.0
    last_activity_change: datetime.datetime = None
    productivity_score: float = 0.0

@dataclass
class ItemCompletionEvent:
    """Record when an item is completed"""
    timestamp: datetime.datetime
    employee_id: str
    item_type: str
    completion_time: float  # Time taken to complete this item
    quality_score: float = 1.0  # Can be enhanced with quality detection

class FactoryMonitoringSystem:
    def __init__(self, db_path="factory_monitoring.db"):
        self.db_path = db_path
        self.setup_database()
        
        # Employee tracking
        self.employee_sessions = {}  # track_id -> EmployeeSession
        self.employee_counter = 0
        
        # Activity tracking
        self.activity_start_times = {}  # track_id -> timestamp
        self.completed_items = []  # List of ItemCompletionEvent
        
        # Performance metrics
        self.session_start_time = datetime.datetime.now()
        self.total_items_produced = 0
        
        # Configuration
        self.completion_keywords = ["Putting Down", "Processing"]  # Activities that indicate item completion
        self.min_handling_time = 2.0  # Minimum time to consider an item "handled"
        
    def setup_database(self):
        """Initialize SQLite database for persistent storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Employee sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS employee_sessions (
                session_id TEXT PRIMARY KEY,
                employee_id TEXT,
                start_time TEXT,
                end_time TEXT,
                total_items INTEGER,
                total_handling_time REAL,
                productivity_score REAL
            )
        ''')
        
        # Item completions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS item_completions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                employee_id TEXT,
                item_type TEXT,
                completion_time REAL,
                quality_score REAL
            )
        ''')
        
        # Activity logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS activity_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                employee_id TEXT,
                activity TEXT,
                duration REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def register_employee(self, track_id: int) -> str:
        """Register a new employee when first detected"""
        self.employee_counter += 1
        employee_id = f"EMP_{self.employee_counter:03d}"
        
        session = EmployeeSession(
            employee_id=employee_id,
            start_time=datetime.datetime.now(),
            last_activity_change=datetime.datetime.now()
        )
        
        self.employee_sessions[track_id] = session
        print(f"🟢 New employee registered: {employee_id}")
        return employee_id
        
    def update_employee_activity(self, track_id: int, new_activity: str):
        """Update employee activity and log changes"""
        if track_id not in self.employee_sessions:
            return
            
        session = self.employee_sessions[track_id]
        current_time = datetime.datetime.now()
        
        # If activity changed, log the previous activity duration
        if new_activity != session.current_activity:
            if session.current_activity != "None" and session.last_activity_change:
                duration = (current_time - session.last_activity_change).total_seconds()
                self.log_activity(session.employee_id, session.current_activity, duration)
                
                # Update total handling time for productive activities
                if session.current_activity in ["Handling", "Processing", "Carrying", "Picking Up", "Putting Down"]:
                    session.total_handling_time += duration
            
            # Check for item completion
            if new_activity in self.completion_keywords and session.current_activity not in ["None", "Touching"]:
                self.record_item_completion(track_id, session)
            
            # Update session
            session.current_activity = new_activity
            session.last_activity_change = current_time
            
    def record_item_completion(self, track_id: int, session: EmployeeSession):
        """Record when an employee completes an item"""
        if session.last_activity_change is None:
            return
            
        completion_time = (datetime.datetime.now() - session.last_activity_change).total_seconds()
        
        # Only count as completion if handling time is reasonable
        if completion_time >= self.min_handling_time:
            session.items_completed += 1
            self.total_items_produced += 1
            
            completion_event = ItemCompletionEvent(
                timestamp=datetime.datetime.now(),
                employee_id=session.employee_id,
                item_type="Standard",  # Can be enhanced with object classification
                completion_time=completion_time
            )
            
            self.completed_items.append(completion_event)
            self.save_item_completion(completion_event)
            
            print(f"✅ Item completed by {session.employee_id} (Total: {session.items_completed})")
            
    def calculate_productivity_score(self, session: EmployeeSession) -> float:
        """Calculate productivity score for an employee"""
        if session.total_handling_time == 0:
            return 0.0
            
        # Simple productivity metric: items per hour of active work
        hours_worked = session.total_handling_time / 3600
        if hours_worked > 0:
            items_per_hour = session.items_completed / hours_worked
            # Normalize to 0-100 scale (assuming 10 items/hour is excellent)
            return min(100.0, (items_per_hour / 10.0) * 100)
        return 0.0
        
    def log_activity(self, employee_id: str, activity: str, duration: float):
        """Log activity to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO activity_logs (timestamp, employee_id, activity, duration)
            VALUES (?, ?, ?, ?)
        ''', (datetime.datetime.now().isoformat(), employee_id, activity, duration))
        
        conn.commit()
        conn.close()
        
    def save_item_completion(self, completion: ItemCompletionEvent):
        """Save item completion to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO item_completions (timestamp, employee_id, item_type, completion_time, quality_score)
            VALUES (?, ?, ?, ?, ?)
        ''', (completion.timestamp.isoformat(), completion.employee_id, 
              completion.item_type, completion.completion_time, completion.quality_score))
        
        conn.commit()
        conn.close()
        
    def get_shift_summary(self) -> Dict:
        """Generate summary report for current shift"""
        summary = {
            "shift_start": self.session_start_time.isoformat(),
            "shift_duration": (datetime.datetime.now() - self.session_start_time).total_seconds() / 3600,
            "total_employees": len(self.employee_sessions),
            "total_items_produced": self.total_items_produced,
            "employees": []
        }
        
        for session in self.employee_sessions.values():
            productivity = self.calculate_productivity_score(session)
            summary["employees"].append({
                "employee_id": session.employee_id,
                "items_completed": session.items_completed,
                "current_activity": session.current_activity,
                "total_handling_time": session.total_handling_time,
                "productivity_score": productivity
            })
            
        return summary
        
    def export_daily_report(self, date: str = None) -> str:
        """Export daily report to JSON file"""
        if date is None:
            date = datetime.datetime.now().strftime("%Y-%m-%d")
            
        report = self.get_shift_summary()
        report["date"] = date
        
        filename = f"factory_report_{date}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
            
        return filename

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
                self.stopped = True
                print("Error grabbing/retrieving frame from VideoCapture.")       
                break

            with self.lock:
                self.ret = ret
                if ret_retrieve:
                    self.frame = frame

            self.ready.set()

        self.cap.release()

def draw_monitoring_overlay(frame, monitoring_system):
    """Draw monitoring information on the frame"""
    overlay = frame.copy()
    
    # Semi-transparent background for info panel
    cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Title
    cv2.putText(frame, "Factory Monitoring System", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Session info
    session_time = (datetime.datetime.now() - monitoring_system.session_start_time).total_seconds() / 3600
    cv2.putText(frame, f"Session Time: {session_time:.1f}h", (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.putText(frame, f"Total Items: {monitoring_system.total_items_produced}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.putText(frame, f"Active Employees: {len(monitoring_system.employee_sessions)}", (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Individual employee info
    y_offset = 120
    for session in monitoring_system.employee_sessions.values():
        productivity = monitoring_system.calculate_productivity_score(session)
        info_text = f"{session.employee_id}: {session.items_completed} items ({productivity:.1f}%)"
        cv2.putText(frame, info_text, (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 20
        
    return frame

def is_point_in_box(point, box, margin=20):
    """Check if a point (x, y) is inside or close to a box (x, y, w, h)."""
    px, py = point
    bx, by, bw, bh = box
    x1 = bx - bw / 2 - margin
    y1 = by - bh / 2 - margin
    x2 = bx + bw / 2 + margin
    y2 = by + bh / 2 + margin
    return x1 <= px <= x2 and y1 <= py <= y2

if __name__ == "__main__":
    # Initialize monitoring system
    monitoring_system = FactoryMonitoringSystem()
    
    # Load YOLO models
    object_model = YOLO("yolo26x.pt") 
    pose_model = YOLO("yolov8n-pose.pt")

    # Video source
    video_url = "test_video.mp4"
    cap = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG)

    # Tracking variables (same as original)
    track_history = defaultdict(lambda: [])
    interaction_history = defaultdict(lambda: [])
    object_interactions = defaultdict(lambda: "None")
    action_history = defaultdict(lambda: deque(maxlen=10)) 
    confirmed_actions = defaultdict(lambda: "None")
    
    # Report generation timer
    last_report_time = time.time()
    report_interval = 300  # Generate report every 5 minutes

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
                
                frame = obj_results.plot()
                
                for box, track_id in zip(yolo_boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y))) 
                    if len(track) > 30: track.pop(0)

                    # Draw trail
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=3)

                    # Register new employees
                    if track_id not in monitoring_system.employee_sessions:
                        monitoring_system.register_employee(track_id)

                    # Draw Action Label with employee ID
                    action = object_interactions[track_id]
                    if action != "None":
                        employee_id = monitoring_system.employee_sessions[track_id].employee_id
                        label = f"{employee_id}: {action}"
                        cv2.putText(frame, label, (int(x - w/2), int(y - h/2) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # --- B. Pose/Hand Detection ---
            pose_results = pose_model(frame, verbose=False)[0]
            
            wrists = []
            if pose_results.keypoints is not None:
                kpts = pose_results.keypoints.xy.cpu().numpy()
                for person in kpts:
                    if len(person) > 10:
                        if person[9][0] > 0 and person[9][1] > 0: wrists.append(person[9])
                        if person[10][0] > 0 and person[10][1] > 0: wrists.append(person[10])
            
            for wx, wy in wrists:
                cv2.circle(frame, (int(wx), int(wy)), 5, (0, 255, 255), -1)

            # --- C. Interaction Logic (same as original) ---
            for box, track_id in zip(yolo_boxes, track_ids):
                is_touching = False
                closest_wrist = None
                min_dist = float('inf')

                for wrist in wrists:
                    if is_point_in_box(wrist, box, margin=15):
                        is_touching = True
                        dist = ((wrist[0] - box[0])**2 + (wrist[1] - box[1])**2)**0.5
                        if dist < min_dist:
                            min_dist = dist
                            closest_wrist = wrist
                
                int_hist = interaction_history[track_id]
                if is_touching and closest_wrist is not None:
                    int_hist.append((box[0], box[1], closest_wrist[0], closest_wrist[1]))
                    if len(int_hist) > 10: int_hist.pop(0)
                else:
                    interaction_history[track_id] = []

                raw_action = "None"

                if is_touching and len(int_hist) >= 5:
                    curr = int_hist[-1]
                    prev = int_hist[-5]
                    
                    obj_dy = curr[1] - prev[1]
                    obj_dx = curr[0] - prev[0]
                    wrist_dy = curr[3] - prev[3]
                    wrist_dx = curr[2] - prev[2]
                    
                    UP_THRESH = -3.0
                    DOWN_THRESH = 3.0
                    MOVE_TOGETHER_TOL = 15
                    
                    if obj_dy < UP_THRESH and wrist_dy < UP_THRESH:
                        diff = abs(obj_dy - wrist_dy)
                        if diff < MOVE_TOGETHER_TOL:
                            raw_action = "Picking Up"
                        else:
                            raw_action = "Handling"
                    elif obj_dy > DOWN_THRESH and wrist_dy > DOWN_THRESH:
                        diff = abs(obj_dy - wrist_dy)
                        if diff < MOVE_TOGETHER_TOL:
                            raw_action = "Putting Down"
                        else:
                            raw_action = "Handling"
                    elif abs(obj_dy) < 2.0 and abs(obj_dx) < 2.0:
                        hand_movement = (wrist_dx**2 + wrist_dy**2)**0.5
                        if hand_movement > 3.0:
                            raw_action = "Processing"
                        else:
                            raw_action = "Holding"
                    elif abs(obj_dx) > 3.0 and abs(wrist_dx) > 3.0:
                        diff = abs(obj_dx - wrist_dx)
                        if diff < MOVE_TOGETHER_TOL:
                            raw_action = "Carrying"
                        else:
                            raw_action = "Handling"
                    else:
                        raw_action = "Handling"
                elif is_touching:
                    raw_action = "Touching"

                action_history[track_id].append(raw_action)
                
                counts = Counter(action_history[track_id])
                if counts:
                    most_common, frequency = counts.most_common(1)[0]
                    
                    if most_common == "None":
                        confirmed_actions[track_id] = "None"
                    elif frequency >= 6:
                        confirmed_actions[track_id] = most_common
                    
                    # Update monitoring system with new activity
                    monitoring_system.update_employee_activity(track_id, confirmed_actions[track_id])
                    object_interactions[track_id] = confirmed_actions[track_id]

            # Draw monitoring overlay
            frame = draw_monitoring_overlay(frame, monitoring_system)
            
            # Generate periodic reports
            current_time = time.time()
            if current_time - last_report_time > report_interval:
                summary = monitoring_system.get_shift_summary()
                print("\n" + "="*50)
                print("SHIFT SUMMARY")
                print("="*50)
                print(f"Total Items Produced: {summary['total_items_produced']}")
                print(f"Active Employees: {summary['total_employees']}")
                for emp in summary['employees']:
                    print(f"  {emp['employee_id']}: {emp['items_completed']} items ({emp['productivity_score']:.1f}% productivity)")
                print("="*50 + "\n")
                last_report_time = current_time

            cv2.imshow("Factory Monitoring System", frame)
            
            # Keyboard shortcuts
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):  # Generate report
                filename = monitoring_system.export_daily_report()
                print(f"Report exported to: {filename}")
            elif key == ord("s"):  # Print summary
                summary = monitoring_system.get_shift_summary()
                print(json.dumps(summary, indent=2))
                
        else:
            break

    # Save final session data
    for session in monitoring_system.employee_sessions.values():
        productivity = monitoring_system.calculate_productivity_score(session)
        print(f"Final stats for {session.employee_id}: {session.items_completed} items, {productivity:.1f}% productivity")

    cap.release()
    cv2.destroyAllWindows()