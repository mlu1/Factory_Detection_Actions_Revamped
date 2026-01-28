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
from orientation_detector import OrientationDetector, Orientation, OrientationChange
from completion_zone_detector import CompletionZoneDetector, CompletionEvent

@dataclass
class BicycleSeatAssemblyEvent:
    """Record bicycle seat assembly completion events"""
    timestamp: datetime.datetime
    employee_id: str
    seat_type: str
    assembly_stage: str
    completion_time: float
    quality_score: float = 1.0
    seat_size: str = "Unknown"
    color: str = "Unknown"
    defects_detected: List[str] = None
    orientation_changes_during_assembly: int = 0  # Track orientation changes during assembly

@dataclass
class EmployeeSession:
    """Track an employee's work session with bicycle seat specific metrics"""
    employee_id: str
    start_time: datetime.datetime
    current_activity: str = "None"
    seats_completed: int = 0
    total_handling_time: float = 0.0
    last_activity_change: datetime.datetime = None
    productivity_score: float = 0.0
    quality_average: float = 0.0
    current_seat_start_time: Optional[datetime.datetime] = None
    assembly_stage: str = "preparation"
    orientation_changes: int = 0  # Track total orientation changes performed

class BicycleSeatFactoryMonitoringSystem:
    def __init__(self, config_path="bicycle_seat_config.json", db_path="bicycle_seat_factory_monitoring.db"):
        self.config = self.load_config(config_path)
        self.db_path = db_path
        self.setup_database()
        
        # Employee tracking
        self.employee_sessions = {}
        self.employee_counter = 0
        
        # Bicycle seat specific tracking
        self.completed_seats = []
        self.current_seat_work = {}  # track_id -> current seat being worked on
        
        # Performance metrics
        self.session_start_time = datetime.datetime.now()
        self.total_seats_produced = 0
        
        # Orientation tracking for bicycle seats
        self.orientation_detector = OrientationDetector(
            history_size=10, 
            confidence_threshold=0.7
        )
        self.task_start_orientations = {}  # track_id -> orientation count when task started
        
        # Completion zone tracking for bicycle seat assembly
        self.completion_zone_detector = CompletionZoneDetector()
        self.zone_completions = []  # List of CompletionEvent from zones
        
        # Activity definitions specific to bicycle seat manufacturing
        self.seat_activities = self.config.get('bicycle_seat_activities', {})
        self.completion_keywords = ["Putting Down", "Processing", "Assembly Complete", "Inspection"]
        self.min_handling_time = self.config['monitoring_config']['min_handling_time_seconds']
        
    def load_config(self, config_path):
        """Load shoe manufacturing specific configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file {config_path} not found, using defaults")
            return self.get_default_config()
            
    def get_default_config(self):
        """Default configuration for bicycle seat manufacturing"""
        return {
            "monitoring_config": {
                "completion_activities": ["Putting Down", "Processing"],
                "min_handling_time_seconds": 4.0,
                "productivity_target_items_per_hour": 6
            },
            "production_targets": {
                "seats_per_hour_per_worker": 6
            },
            "quality_metrics": {
                "overall_craftsmanship": 0.9
            }
        }
        
    def setup_database(self):
        """Initialize SQLite database for bicycle seat manufacturing data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Employee sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS employee_sessions (
                session_id TEXT PRIMARY KEY,
                employee_id TEXT,
                start_time TEXT,
                end_time TEXT,
                total_seats INTEGER,
                total_handling_time REAL,
                productivity_score REAL,
                quality_average REAL
            )
        ''')
        
        # Bicycle seat assembly completions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS seat_completions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                employee_id TEXT,
                seat_type TEXT,
                assembly_stage TEXT,
                completion_time REAL,
                quality_score REAL,
                seat_size TEXT,
                color TEXT,
                defects TEXT
            )
        ''')
        
        # Activity logs table (enhanced for bicycle seat manufacturing)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS activity_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                employee_id TEXT,
                activity TEXT,
                assembly_stage TEXT,
                duration REAL,
                seat_component TEXT
            )
        ''')
        
        # Quality control table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quality_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                employee_id TEXT,
                seat_id TEXT,
                quality_score REAL,
                defects_found TEXT,
                passed_qc BOOLEAN
            )
        ''')
        
        # Orientation changes table for bicycle seat assembly
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orientation_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                employee_id TEXT,
                track_id INTEGER,
                previous_orientation TEXT,
                new_orientation TEXT,
                confidence REAL,
                assembly_stage TEXT,
                during_task TEXT
            )
        ''')
        
        # Completion zone events table for bicycle seat assembly
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS completion_zone_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                employee_id TEXT,
                track_id INTEGER,
                zone_name TEXT,
                time_to_complete REAL,
                assembly_stage TEXT,
                bbox_x REAL,
                bbox_y REAL,
                bbox_w REAL,
                bbox_h REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def register_employee(self, track_id: int) -> str:
        """Register a new employee when first detected"""
        self.employee_counter += 1
        employee_id = f"SEAT_WORKER_{self.employee_counter:03d}"
        
        session = EmployeeSession(
            employee_id=employee_id,
            start_time=datetime.datetime.now(),
            last_activity_change=datetime.datetime.now()
        )
        
        self.employee_sessions[track_id] = session
        print(f"🚴 New bicycle seat worker registered: {employee_id}")
        return employee_id
        
    def detect_assembly_stage(self, activity: str, duration: float) -> str:
        """Detect current assembly stage based on activity pattern"""
        if activity in ["Cutting", "Trimming"]:
            return "cutting"
        elif "Padding" in activity or activity == "Processing" and duration < 8:
            return "padding_attachment"
        elif "Cover" in activity or "Gluing" in activity:
            return "cover_installation"
        elif "Rail" in activity or "Pressing" in activity:
            return "rail_mounting"
        elif activity == "Processing" and duration > 10:
            return "finishing"
        elif activity == "Inspection":
            return "quality_check"
        else:
            return "general_assembly"
            
    def update_employee_activity(self, track_id: int, new_activity: str):
        """Update employee activity with shoe manufacturing specifics"""
        if track_id not in self.employee_sessions:
            return
            
        session = self.employee_sessions[track_id]
        current_time = datetime.datetime.now()
        
        # Track task start for orientation counting
        if new_activity != "None" and session.current_activity == "None":
            # Starting a new task - record current orientation count
            self.task_start_orientations[track_id] = session.orientation_changes
        
        # If activity changed, log the previous activity
        if new_activity != session.current_activity:
            if session.current_activity != "None" and session.last_activity_change:
                duration = (current_time - session.last_activity_change).total_seconds()
                
                # Detect assembly stage
                assembly_stage = self.detect_assembly_stage(session.current_activity, duration)
                
                self.log_seat_activity(session.employee_id, session.current_activity, 
                                     assembly_stage, duration)
                
                # Update total handling time for productive activities
                if session.current_activity in ["Handling", "Processing", "Carrying", 
                                              "Picking Up", "Putting Down", "Padding Attachment", "Cover Installation"]:
                    session.total_handling_time += duration
            
            # Check for bicycle seat completion
            if new_activity in self.completion_keywords and session.current_activity not in ["None", "Touching"]:
                self.record_seat_completion(track_id, session)
            
            # Update session
            session.current_activity = new_activity
            session.last_activity_change = current_time
            
    def record_seat_completion(self, track_id: int, session: EmployeeSession):
        """Record when a worker completes a bicycle seat"""
        if session.last_activity_change is None:
            return
            
        completion_time = (datetime.datetime.now() - session.last_activity_change).total_seconds()
        
        # Only count as completion if handling time is reasonable for bicycle seat work
        if completion_time >= self.min_handling_time:
            # Calculate orientation changes during this assembly
            task_start_count = self.task_start_orientations.get(track_id, session.orientation_changes)
            orientation_changes_during_assembly = session.orientation_changes - task_start_count
            
            session.seats_completed += 1
            self.total_seats_produced += 1
            
            # Estimate quality score based on completion time and consistency
            target_time = 3600 / self.config['production_targets']['seats_per_hour_per_worker']
            quality_score = max(0.5, min(1.0, target_time / completion_time))
            
            completion_event = BicycleSeatAssemblyEvent(
                timestamp=datetime.datetime.now(),
                employee_id=session.employee_id,
                seat_type="Standard",
                assembly_stage=session.assembly_stage,
                completion_time=completion_time,
                quality_score=quality_score,
                orientation_changes_during_assembly=orientation_changes_during_assembly
            )
            
            self.completed_seats.append(completion_event)
            self.save_seat_completion(completion_event)
            
            print(f"🚴 Seat completed by {session.employee_id} (Total: {session.seats_completed}, Quality: {quality_score:.2f}, Orientations: {orientation_changes_during_assembly})")
            
    def calculate_seat_productivity_score(self, session: EmployeeSession) -> float:
        """Calculate productivity score specific to bicycle seat manufacturing"""
        if session.total_handling_time == 0:
            return 0.0
            
        hours_worked = session.total_handling_time / 3600
        if hours_worked > 0:
            seats_per_hour = session.seats_completed / hours_worked
            target_seats_per_hour = self.config['production_targets']['seats_per_hour_per_worker']
            return min(100.0, (seats_per_hour / target_seats_per_hour) * 100)
        return 0.0
        
    def log_seat_activity(self, employee_id: str, activity: str, assembly_stage: str, duration: float):
        """Log bicycle seat manufacturing activity to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO activity_logs (timestamp, employee_id, activity, assembly_stage, duration, seat_component)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (datetime.datetime.now().isoformat(), employee_id, activity, 
              assembly_stage, duration, "general"))
        
        conn.commit()
        conn.close()
        
    def save_seat_completion(self, completion: BicycleSeatAssemblyEvent):
        """Save bicycle seat completion to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        defects_str = json.dumps(completion.defects_detected) if completion.defects_detected else None
        
        cursor.execute('''
            INSERT INTO seat_completions (timestamp, employee_id, seat_type, assembly_stage, 
                                        completion_time, quality_score, seat_size, color, defects)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (completion.timestamp.isoformat(), completion.employee_id, completion.seat_type,
              completion.assembly_stage, completion.completion_time, completion.quality_score,
              completion.seat_size, completion.color, defects_str))
        
        conn.commit()
        conn.close()
        
    def get_production_summary(self) -> Dict:
        """Generate bicycle seat manufacturing summary report"""
        summary = {
            "shift_start": self.session_start_time.isoformat(),
            "shift_duration": (datetime.datetime.now() - self.session_start_time).total_seconds() / 3600,
            "total_employees": len(self.employee_sessions),
            "total_seats_produced": self.total_seats_produced,
            "average_quality": sum([event.quality_score for event in self.completed_seats]) / max(1, len(self.completed_seats)),
            "production_rate": self.total_seats_produced / max(1, (datetime.datetime.now() - self.session_start_time).total_seconds() / 3600),
            "employees": []
        }
        
        for session in self.employee_sessions.values():
            productivity = self.calculate_seat_productivity_score(session)
            summary["employees"].append({
                "employee_id": session.employee_id,
                "seats_completed": session.seats_completed,
                "current_activity": session.current_activity,
                "assembly_stage": session.assembly_stage,
                "total_handling_time": session.total_handling_time,
                "productivity_score": productivity,
                "quality_average": session.quality_average
            })
            
        return summary
    
    def process_orientation_change(self, orientation_change: OrientationChange):
        """Process and log orientation changes in bicycle seat assembly"""
        if orientation_change.track_id in self.employee_sessions:
            session = self.employee_sessions[orientation_change.track_id]
            session.orientation_changes += 1
            
            # Log to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO orientation_changes 
                (timestamp, employee_id, track_id, previous_orientation, new_orientation, 
                 confidence, assembly_stage, during_task)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                orientation_change.timestamp.isoformat(),
                session.employee_id,
                orientation_change.track_id,
                orientation_change.previous_orientation.value,
                orientation_change.new_orientation.value,
                orientation_change.confidence,
                session.assembly_stage,
                session.current_activity
            ))
            
            conn.commit()
            conn.close()
            
            print(f"🔄 {session.employee_id} ({session.assembly_stage}): {orientation_change.previous_orientation.value} → {orientation_change.new_orientation.value}")
    
    def update_object_tracking(self, track_id: int, bbox: tuple, frame: np.ndarray):
        """Update object tracking with orientation and completion zone detection for bicycle seat assembly"""
        # Update orientation detection
        orientation_change = self.orientation_detector.update_orientation(track_id, bbox, frame)
        
        # Process any orientation changes
        if orientation_change:
            self.process_orientation_change(orientation_change)
        
        # Update completion zone detection
        completion_event = self.completion_zone_detector.update_object_tracking(
            track_id, bbox, self.employee_sessions, datetime.datetime.now()
        )
        
        # Process any completion zone events
        if completion_event:
            self.process_completion_zone_event(completion_event)
    
    def process_completion_zone_event(self, completion_event: CompletionEvent):
        """Process completion zone events for bicycle seat assembly"""
        if completion_event.track_id in self.employee_sessions:
            session = self.employee_sessions[completion_event.track_id]
            
            # Only count edge completions as actual bicycle seat completions
            if completion_event.zone_name == "Edge_Completion":
                # Calculate orientation changes during this assembly
                task_start_count = self.task_start_orientations.get(completion_event.track_id, session.orientation_changes)
                orientation_changes_during_assembly = session.orientation_changes - task_start_count
                
                session.seats_completed += 1
                self.total_seats_produced += 1
                
                # Estimate quality score based on completion time
                target_time = 3600 / self.config['production_targets']['seats_per_hour_per_worker']
                quality_score = max(0.5, min(1.0, target_time / completion_event.time_to_complete))
                
                # Create enhanced completion event
                seat_completion = BicycleSeatAssemblyEvent(
                    timestamp=completion_event.timestamp,
                    employee_id=completion_event.employee_id,
                    seat_type="Standard",
                    assembly_stage=session.assembly_stage,
                    completion_time=completion_event.time_to_complete,
                    quality_score=quality_score,
                    orientation_changes_during_assembly=orientation_changes_during_assembly
                )
                
                self.completed_seats.append(seat_completion)
                self.save_seat_completion(seat_completion)
                
                print(f"🚴 {completion_event.employee_id} completed bicycle seat by placing in {completion_event.zone_name} (Time: {completion_event.time_to_complete:.1f}s, Quality: {quality_score:.2f}, Orientations: {orientation_changes_during_assembly})")
            
            # Log to database
            self.save_completion_zone_event(completion_event)
            
            # Store in our list
            self.zone_completions.append(completion_event)
    
    def save_completion_zone_event(self, completion_event: CompletionEvent):
        """Save completion zone event to database"""
        if completion_event.track_id in self.employee_sessions:
            session = self.employee_sessions[completion_event.track_id]
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            x, y, w, h = completion_event.bbox
            cursor.execute('''
                INSERT INTO completion_zone_events 
                (timestamp, employee_id, track_id, zone_name, time_to_complete, 
                 assembly_stage, bbox_x, bbox_y, bbox_w, bbox_h)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                completion_event.timestamp.isoformat(),
                completion_event.employee_id,
                completion_event.track_id,
                completion_event.zone_name,
                completion_event.time_to_complete,
                session.assembly_stage,
                x, y, w, h
            ))
            
            conn.commit()
            conn.close()
    
    def get_orientation_summary(self) -> Dict:
        """Get summary of orientation changes in bicycle seat assembly"""
        total_changes = sum(session.orientation_changes for session in self.employee_sessions.values())
        
        # Get recent changes (last hour)
        one_hour_ago = datetime.datetime.now() - datetime.timedelta(hours=1)
        recent_changes = self.orientation_detector.get_orientation_changes(since=one_hour_ago)
        
        summary = {
            "total_orientation_changes": total_changes,
            "recent_changes_last_hour": len(recent_changes),
            "active_orientations": {},
            "assembly_stage_orientations": {}
        }
        
        # Get current orientations for active objects
        for track_id, session in self.employee_sessions.items():
            orientation = self.orientation_detector.get_current_orientation(track_id)
            summary["active_orientations"][session.employee_id] = orientation.value
            
            # Group by assembly stage
            stage = session.assembly_stage
            if stage not in summary["assembly_stage_orientations"]:
                summary["assembly_stage_orientations"][stage] = []
            summary["assembly_stage_orientations"][stage].append({
                "employee": session.employee_id,
                "orientation": orientation.value,
                "orientation_changes": session.orientation_changes
            })
            
        return summary

def draw_bicycle_seat_monitoring_overlay(frame, monitoring_system):
    """Draw bicycle seat manufacturing specific monitoring information"""
    overlay = frame.copy()
    
    # Larger semi-transparent background for info panel
    cv2.rectangle(overlay, (10, 10), (550, 240), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Title
    cv2.putText(frame, "Bicycle Seat Factory Monitor", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Session info
    session_time = (datetime.datetime.now() - monitoring_system.session_start_time).total_seconds() / 3600
    cv2.putText(frame, f"Session Time: {session_time:.1f}h", (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.putText(frame, f"Bicycle Seats: {monitoring_system.total_seats_produced}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.putText(frame, f"Workers: {len(monitoring_system.employee_sessions)}", (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Production rate
    hours = max(0.1, session_time)
    rate = monitoring_system.total_seats_produced / hours
    cv2.putText(frame, f"Rate: {rate:.1f} seats/hour", (20, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Orientation changes summary
    total_orientation_changes = sum(session.orientation_changes for session in monitoring_system.employee_sessions.values())
    cv2.putText(frame, f"Orientation Changes: {total_orientation_changes}", (20, 140), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
    
    # Completion zone summary
    zone_counts = monitoring_system.completion_zone_detector.get_zone_counts()
    edge_count = zone_counts.get("Edge_Completion", 0)
    cv2.putText(frame, f"Seats in Completion: {edge_count}", (20, 160), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Individual worker info
    y_offset = 180
    for session in monitoring_system.employee_sessions.values():
        productivity = monitoring_system.calculate_seat_productivity_score(session)
        info_text = f"{session.employee_id}: {session.seats_completed} seats ({productivity:.1f}%, {session.orientation_changes} orient)"
        cv2.putText(frame, info_text, (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 18
        
    return frame

# Import the main monitoring logic from the original file
from factory_monitor_enhanced import (
    VideoStream, is_point_in_box
)

if __name__ == "__main__":
    # Initialize shoe factory monitoring system
    monitoring_system = BicycleSeatFactoryMonitoringSystem()
    
    # Load YOLO models
    object_model = YOLO("yolo26x.pt") 
    pose_model = YOLO("yolov8n-pose.pt")

    # Video source - modify this to use your video file
    video_url = "TEST_VID_DATA.mp4"  # Change this to your video filename
    # Examples:
    # video_url = "my_bicycle_factory.mp4"
    # video_url = "C:/Users/user/Videos/factory_video.mp4"  # Full path
    # video_url = 0  # Use webcam
    cap = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG)

    # Tracking variables
    track_history = defaultdict(lambda: [])
    interaction_history = defaultdict(lambda: [])
    object_interactions = defaultdict(lambda: "None")
    action_history = defaultdict(lambda: deque(maxlen=12))  # Slightly longer for shoe work
    confirmed_actions = defaultdict(lambda: "None")
    
    # Report generation timer
    last_report_time = time.time()
    report_interval = monitoring_system.config['monitoring_config']['report_interval_seconds']

    print("🥿 Starting Shoe Factory Monitoring System...")
    print("Activities tracked: Cutting, Stitching, Sole Attachment, Finishing, Quality Check")

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # Auto-configure completion zones based on frame size (first frame only)
            if frame is not None and len(monitoring_system.completion_zone_detector.completion_zones) == 0:
                h, w = frame.shape[:2]
                monitoring_system.completion_zone_detector.configure_zones_from_frame(w, h)
            
            # Object Detection & Tracking
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
                    if len(track) > 35: track.pop(0)  # Longer history for bicycle seat work

                    # Draw trail
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=3)

                    # Register new workers
                    if track_id not in monitoring_system.employee_sessions:
                        monitoring_system.register_employee(track_id)

                    # Update orientation tracking for this object
                    monitoring_system.update_object_tracking(track_id, (x, y, w, h), frame)
                    
                    # Draw orientation info on the frame
                    frame = monitoring_system.orientation_detector.draw_orientation_info(frame, (x, y, w, h), track_id)

                    # Draw labels with worker ID, activity, and orientation
                    action = object_interactions[track_id]
                    if action != "None":
                        employee_id = monitoring_system.employee_sessions[track_id].employee_id
                        orientation = monitoring_system.orientation_detector.get_current_orientation(track_id)
                        assembly_stage = monitoring_system.employee_sessions[track_id].assembly_stage
                        label = f"{employee_id}: {action} | {orientation.value} | {assembly_stage}"
                        cv2.putText(frame, label, (int(x - w/2), int(y - h/2) - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                
                # Clean up orientation tracking for inactive objects
                monitoring_system.orientation_detector.cleanup_old_tracks(track_ids)
                monitoring_system.completion_zone_detector.cleanup_old_tracks(track_ids)
            
            # Draw completion zones
            frame = monitoring_system.completion_zone_detector.draw_zones(frame)
                        worker_id = monitoring_system.employee_sessions[track_id].employee_id
                        stage = monitoring_system.employee_sessions[track_id].assembly_stage
                        label = f"{worker_id}: {action} ({stage})"
                        cv2.putText(frame, label, (int(x - w/2), int(y - h/2) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Pose/Hand Detection (same as original)
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

            # Enhanced Interaction Logic for shoe manufacturing
            for box, track_id in zip(yolo_boxes, track_ids):
                is_touching = False
                closest_wrist = None
                min_dist = float('inf')

                margin = monitoring_system.config['interaction_thresholds']['wrist_detection_margin']
                for wrist in wrists:
                    if is_point_in_box(wrist, box, margin=margin):
                        is_touching = True
                        dist = ((wrist[0] - box[0])**2 + (wrist[1] - box[1])**2)**0.5
                        if dist < min_dist:
                            min_dist = dist
                            closest_wrist = wrist
                
                int_hist = interaction_history[track_id]
                if is_touching and closest_wrist is not None:
                    int_hist.append((box[0], box[1], closest_wrist[0], closest_wrist[1]))
                    if len(int_hist) > 12: int_hist.pop(0)  # Longer history for detailed shoe work
                else:
                    interaction_history[track_id] = []

                raw_action = "None"

                if is_touching and len(int_hist) >= 6:  # Need more history for bicycle seat work
                    curr = int_hist[-1]
                    prev = int_hist[-6]  # 6 frames ago for more stable detection
                    
                    obj_dy = curr[1] - prev[1]
                    obj_dx = curr[0] - prev[0]
                    wrist_dy = curr[3] - prev[3]
                    wrist_dx = curr[2] - prev[2]
                    
                    # Adjusted thresholds for bicycle seat manufacturing precision
                    UP_THRESH = monitoring_system.config['interaction_thresholds']['up_movement_threshold']
                    DOWN_THRESH = monitoring_system.config['interaction_thresholds']['down_movement_threshold']
                    MOVE_TOGETHER_TOL = monitoring_system.config['interaction_thresholds']['movement_tolerance']
                    
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
                        if hand_movement > 2.5:  # More sensitive for detailed bicycle seat work
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
                    
                    # Higher consistency requirement for bicycle seat manufacturing
                    consistency_threshold = monitoring_system.config['detection_config']['consistency_threshold']
                    if most_common == "None":
                        confirmed_actions[track_id] = "None"
                    elif frequency >= consistency_threshold:
                        confirmed_actions[track_id] = most_common
                    
                    # Update monitoring system
                    monitoring_system.update_employee_activity(track_id, confirmed_actions[track_id])
                    object_interactions[track_id] = confirmed_actions[track_id]

            # Draw bicycle seat manufacturing specific overlay
            frame = draw_bicycle_seat_monitoring_overlay(frame, monitoring_system)
            
            # Generate periodic reports
            current_time = time.time()
            if current_time - last_report_time > report_interval:
                summary = monitoring_system.get_production_summary()
                print("\n" + "="*60)
                print("� BICYCLE SEAT PRODUCTION SUMMARY")
                print("="*60)
                print(f"Total Bicycle Seats: {summary['total_seats_produced']}")
                print(f"Production Rate: {summary['production_rate']:.1f} seats/hour")
                print(f"Average Quality: {summary['average_quality']:.2f}")
                print(f"Active Workers: {summary['total_employees']}")
                for emp in summary['employees']:
                    print(f"  {emp['employee_id']}: {emp['seats_completed']} seats ({emp['productivity_score']:.1f}% productivity)")
                print("="*60 + "\n")
                last_report_time = current_time

            cv2.imshow("Bicycle Seat Factory Monitoring System", frame)
            
            # Keyboard shortcuts
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):  # Generate report
                summary = monitoring_system.get_production_summary()
                filename = f"bicycle_seat_production_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w') as f:
                    json.dump(summary, f, indent=2)
                print(f"🚴 Bicycle seat production report exported to: {filename}")
            elif key == ord("s"):  # Print summary
                summary = monitoring_system.get_production_summary()
                print(json.dumps(summary, indent=2))
            elif key == ord("o"):  # Print orientation summary
                orientation_summary = monitoring_system.get_orientation_summary()
                print("\n" + "="*50)
                print("🔄 BICYCLE SEAT ORIENTATION SUMMARY")
                print("="*50)
                print(json.dumps(orientation_summary, indent=2))
                print("="*50 + "\n")
                
        else:
            break

    # Final session summary
    print("\n" + "="*60)
    print("� FINAL SHIFT SUMMARY")
    print("="*60)
    for session in monitoring_system.employee_sessions.values():
        productivity = monitoring_system.calculate_seat_productivity_score(session)
        print(f"{session.employee_id}: {session.seats_completed} bicycle seats completed ({productivity:.1f}% productivity, {session.orientation_changes} orientation changes)")
    print("="*60)

    cap.release()
    cv2.destroyAllWindows()