"""
Completion Zone Detection Module
Detects when objects are placed in designated completion areas (edge of table)
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import datetime
from collections import defaultdict, deque

@dataclass
class CompletionZone:
    """Define a completion area on the workstation"""
    name: str
    x: int  # Top-left x coordinate
    y: int  # Top-left y coordinate 
    width: int
    height: int
    color: Tuple[int, int, int] = (0, 255, 0)  # Green by default
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is within this completion zone"""
        return (self.x <= x <= self.x + self.width and 
                self.y <= y <= self.y + self.height)
    
    def contains_center(self, bbox: Tuple[float, float, float, float]) -> bool:
        """Check if the center of a bounding box is in this zone"""
        center_x, center_y = bbox[0], bbox[1]
        return self.contains_point(center_x, center_y)
    
    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw the completion zone on the frame"""
        cv2.rectangle(frame, 
                     (self.x, self.y), 
                     (self.x + self.width, self.y + self.height), 
                     self.color, 2)
        
        # Add label
        cv2.putText(frame, self.name, 
                   (self.x + 5, self.y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color, 2)
        return frame

@dataclass
class CompletionEvent:
    """Record when an object enters a completion zone"""
    timestamp: datetime.datetime
    track_id: int
    employee_id: str
    zone_name: str
    bbox: Tuple[float, float, float, float]
    time_to_complete: float  # Time from task start to completion

class CompletionZoneDetector:
    def __init__(self):
        """Initialize completion zone detector"""
        self.completion_zones: Dict[str, CompletionZone] = {}
        self.objects_in_zones: Dict[str, Set[int]] = defaultdict(set)  # zone_name -> set of track_ids
        self.completion_events: List[CompletionEvent] = []
        
        # Track object history to detect when they enter zones
        self.object_zone_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=5))
        self.objects_previously_in_work_area: Set[int] = set()
        
        # Configure default zones
        self.setup_default_zones()
    
    def setup_default_zones(self):
        """Setup default completion zones for bicycle seat manufacturing"""
        # Edge completion zone (right edge of table)
        self.add_completion_zone(
            name="Edge_Completion", 
            x=500, y=100, width=140, height=300,  # Adjust based on your video resolution
            color=(0, 255, 0)  # Green
        )
        
        # Quality check zone (optional)
        self.add_completion_zone(
            name="Quality_Check", 
            x=300, y=50, width=100, height=100,
            color=(0, 255, 255)  # Yellow
        )
    
    def add_completion_zone(self, name: str, x: int, y: int, width: int, height: int, 
                           color: Tuple[int, int, int] = (0, 255, 0)):
        """Add a new completion zone"""
        zone = CompletionZone(name, x, y, width, height, color)
        self.completion_zones[name] = zone
        self.objects_in_zones[name] = set()
        print(f"📍 Added completion zone: {name} at ({x},{y}) size {width}x{height}")
    
    def configure_zones_from_frame(self, frame_width: int, frame_height: int):
        """Auto-configure zones based on frame dimensions"""
        # Clear existing zones
        self.completion_zones.clear()
        self.objects_in_zones.clear()
        
        # Edge completion zone (right 20% of frame)
        edge_width = int(frame_width * 0.2)
        edge_x = frame_width - edge_width - 10
        
        self.add_completion_zone(
            name="Edge_Completion",
            x=edge_x, y=int(frame_height * 0.2), 
            width=edge_width, height=int(frame_height * 0.6),
            color=(0, 255, 0)
        )
        
        # Quality check zone (top center)
        self.add_completion_zone(
            name="Quality_Check",
            x=int(frame_width * 0.4), y=10,
            width=int(frame_width * 0.2), height=int(frame_height * 0.15),
            color=(0, 255, 255)
        )
        
        print(f"🎯 Auto-configured zones for {frame_width}x{frame_height} frame")
    
    def update_object_tracking(self, track_id: int, bbox: Tuple[float, float, float, float], 
                              employee_sessions: Dict, current_time: datetime.datetime) -> Optional[CompletionEvent]:
        """
        Update object tracking and detect completion events
        
        Returns:
            CompletionEvent if object completed, None otherwise
        """
        x, y, w, h = bbox
        
        # Check which zones this object is currently in
        current_zones = set()
        for zone_name, zone in self.completion_zones.items():
            if zone.contains_center(bbox):
                current_zones.add(zone_name)
        
        # Update zone history
        self.object_zone_history[track_id].append(current_zones)
        
        # Check for completion events
        completion_event = None
        
        # Look for objects that just entered the completion zone
        for zone_name in current_zones:
            if track_id not in self.objects_in_zones[zone_name]:
                # Object just entered this completion zone
                if zone_name == "Edge_Completion" and track_id in self.objects_previously_in_work_area:
                    # This is a completion event!
                    if track_id in employee_sessions:
                        session = employee_sessions[track_id]
                        
                        # Calculate time to complete (from last activity change)
                        time_to_complete = 0.0
                        if session.last_activity_change:
                            time_to_complete = (current_time - session.last_activity_change).total_seconds()
                        
                        completion_event = CompletionEvent(
                            timestamp=current_time,
                            track_id=track_id,
                            employee_id=session.employee_id,
                            zone_name=zone_name,
                            bbox=bbox,
                            time_to_complete=time_to_complete
                        )
                        
                        self.completion_events.append(completion_event)
                        print(f"✅ {session.employee_id} completed bicycle seat (placed in {zone_name})")
                
                # Add to zone tracking
                self.objects_in_zones[zone_name].add(track_id)
        
        # Remove from zones that object is no longer in
        for zone_name in self.completion_zones.keys():
            if zone_name not in current_zones and track_id in self.objects_in_zones[zone_name]:
                self.objects_in_zones[zone_name].remove(track_id)
        
        # Track that this object was in the work area
        if not current_zones or "Edge_Completion" not in current_zones:
            self.objects_previously_in_work_area.add(track_id)
        
        return completion_event
    
    def get_zone_counts(self) -> Dict[str, int]:
        """Get count of objects currently in each zone"""
        return {zone_name: len(objects) for zone_name, objects in self.objects_in_zones.items()}
    
    def get_completion_summary(self, since: Optional[datetime.datetime] = None) -> Dict:
        """Get summary of completion events"""
        events = self.completion_events
        if since:
            events = [e for e in events if e.timestamp >= since]
        
        summary = {
            "total_completions": len(events),
            "completion_by_zone": {},
            "completion_by_employee": {},
            "average_completion_time": 0.0
        }
        
        # Group by zone
        for event in events:
            zone = event.zone_name
            summary["completion_by_zone"][zone] = summary["completion_by_zone"].get(zone, 0) + 1
        
        # Group by employee
        for event in events:
            emp = event.employee_id
            summary["completion_by_employee"][emp] = summary["completion_by_employee"].get(emp, 0) + 1
        
        # Calculate average completion time
        if events:
            total_time = sum(e.time_to_complete for e in events)
            summary["average_completion_time"] = total_time / len(events)
        
        return summary
    
    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """Draw all completion zones on the frame"""
        for zone in self.completion_zones.values():
            frame = zone.draw(frame)
        
        # Draw zone counts
        y_offset = 30
        for zone_name, count in self.get_zone_counts().items():
            text = f"{zone_name}: {count} items"
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            y_offset += 25
        
        return frame
    
    def cleanup_old_tracks(self, active_track_ids: List[int]):
        """Remove tracking data for inactive objects"""
        active_set = set(active_track_ids)
        
        # Clean up zone tracking
        for zone_name in self.objects_in_zones:
            self.objects_in_zones[zone_name] &= active_set
        
        # Clean up history
        inactive_tracks = set(self.object_zone_history.keys()) - active_set
        for track_id in inactive_tracks:
            del self.object_zone_history[track_id]
        
        # Clean up work area tracking
        self.objects_previously_in_work_area &= active_set
    
    def is_in_completion_zone(self, track_id: int, zone_name: str = "Edge_Completion") -> bool:
        """Check if an object is currently in a specific completion zone"""
        return track_id in self.objects_in_zones.get(zone_name, set())