"""
Object Orientation Detection Module
Detects orientation changes: upright, flat, upside down
"""

import cv2
import numpy as np
from enum import Enum
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import datetime
from collections import deque

class Orientation(Enum):
    """Object orientation states"""
    UPRIGHT = "upright"
    FLAT = "flat" 
    UPSIDE_DOWN = "upside_down"
    UNKNOWN = "unknown"

@dataclass
class OrientationChange:
    """Record of orientation change event"""
    timestamp: datetime.datetime
    track_id: int
    previous_orientation: Orientation
    new_orientation: Orientation
    confidence: float
    bounding_box: Tuple[float, float, float, float]  # x, y, w, h

class OrientationDetector:
    def __init__(self, history_size: int = 10, confidence_threshold: float = 0.7):
        """
        Initialize orientation detector
        
        Args:
            history_size: Number of frames to use for orientation smoothing
            confidence_threshold: Minimum confidence to confirm orientation change
        """
        self.history_size = history_size
        self.confidence_threshold = confidence_threshold
        
        # Track orientation history for each object
        self.orientation_history: Dict[int, deque] = {}
        self.current_orientations: Dict[int, Orientation] = {}
        self.orientation_changes: List[OrientationChange] = []
        
        # Track previous bounding boxes for comparison
        self.previous_boxes: Dict[int, Tuple[float, float, float, float]] = {}
        
    def detect_orientation(self, bbox: Tuple[float, float, float, float], 
                          track_id: int, frame: np.ndarray) -> Orientation:
        """
        Detect object orientation based on bounding box dimensions and position
        
        Args:
            bbox: Bounding box (x, y, w, h)
            track_id: Object tracking ID
            frame: Current frame (for advanced analysis if needed)
            
        Returns:
            Detected orientation
        """
        x, y, w, h = bbox
        
        # Calculate aspect ratio
        aspect_ratio = w / h if h > 0 else 1.0
        
        # Method 1: Aspect ratio analysis
        orientation = self._analyze_aspect_ratio(aspect_ratio)
        
        # Method 2: Position change analysis (if we have previous data)
        if track_id in self.previous_boxes:
            prev_x, prev_y, prev_w, prev_h = self.previous_boxes[track_id]
            position_orientation = self._analyze_position_change(
                (prev_x, prev_y, prev_w, prev_h), (x, y, w, h)
            )
            
            # Combine both methods with weights
            orientation = self._combine_orientation_methods(orientation, position_orientation)
        
        # Method 3: Advanced edge analysis (optional enhancement)
        # orientation = self._analyze_object_edges(frame, bbox, orientation)
        
        # Store current box for next frame comparison
        self.previous_boxes[track_id] = bbox
        
        return orientation
    
    def _analyze_aspect_ratio(self, aspect_ratio: float) -> Orientation:
        """
        Determine orientation based on aspect ratio
        
        Assumptions:
        - Upright objects tend to have height > width (aspect ratio < 1)
        - Flat objects tend to have width > height (aspect ratio > 1.5)
        - Upside down detection requires additional context
        """
        if aspect_ratio < 0.6:  # Tall and narrow
            return Orientation.UPRIGHT
        elif aspect_ratio > 1.8:  # Wide and flat
            return Orientation.FLAT
        elif 1.2 < aspect_ratio <= 1.8:  # Moderately wide
            return Orientation.FLAT
        else:  # Square-ish or slightly rectangular
            return Orientation.UNKNOWN
    
    def _analyze_position_change(self, prev_bbox: Tuple[float, float, float, float],
                               curr_bbox: Tuple[float, float, float, float]) -> Orientation:
        """
        Analyze orientation based on how the bounding box changed
        """
        prev_x, prev_y, prev_w, prev_h = prev_bbox
        curr_x, curr_y, curr_w, curr_h = curr_bbox
        
        # Calculate dimension changes
        width_change = abs(curr_w - prev_w)
        height_change = abs(curr_h - prev_h)
        
        # If width increased significantly while height decreased
        if width_change > 20 and height_change > 10:
            if curr_w > prev_w and curr_h < prev_h:
                return Orientation.FLAT
            elif curr_w < prev_w and curr_h > prev_h:
                return Orientation.UPRIGHT
        
        return Orientation.UNKNOWN
    
    def _analyze_object_edges(self, frame: np.ndarray, 
                            bbox: Tuple[float, float, float, float],
                            current_orientation: Orientation) -> Orientation:
        """
        Advanced edge analysis for better orientation detection
        This can be enhanced based on specific object characteristics
        """
        x, y, w, h = bbox
        
        # Extract object region
        x1 = int(max(0, x - w/2))
        y1 = int(max(0, y - h/2))
        x2 = int(min(frame.shape[1], x + w/2))
        y2 = int(min(frame.shape[0], y + h/2))
        
        if x2 > x1 and y2 > y1:
            roi = frame[y1:y2, x1:x2]
            
            # Convert to grayscale and detect edges
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi
                
            edges = cv2.Canny(gray, 50, 150)
            
            # Analyze edge distribution
            horizontal_edges = np.sum(edges, axis=1)  # Sum along width
            vertical_edges = np.sum(edges, axis=0)    # Sum along height
            
            h_variance = np.var(horizontal_edges)
            v_variance = np.var(vertical_edges)
            
            # Objects lying flat often have more uniform horizontal edges
            # Upright objects often have more varied vertical edges
            if h_variance > v_variance * 1.5:
                return Orientation.UPRIGHT
            elif v_variance > h_variance * 1.5:
                return Orientation.FLAT
        
        return current_orientation
    
    def _combine_orientation_methods(self, method1: Orientation, 
                                   method2: Orientation) -> Orientation:
        """Combine results from multiple detection methods"""
        if method1 == method2:
            return method1
        elif method1 != Orientation.UNKNOWN:
            return method1
        elif method2 != Orientation.UNKNOWN:
            return method2
        else:
            return Orientation.UNKNOWN
    
    def update_orientation(self, track_id: int, bbox: Tuple[float, float, float, float], 
                          frame: np.ndarray) -> Optional[OrientationChange]:
        """
        Update orientation for a tracked object and detect changes
        
        Returns:
            OrientationChange if orientation changed, None otherwise
        """
        # Initialize history for new objects
        if track_id not in self.orientation_history:
            self.orientation_history[track_id] = deque(maxlen=self.history_size)
            self.current_orientations[track_id] = Orientation.UNKNOWN
        
        # Detect current orientation
        detected_orientation = self.detect_orientation(bbox, track_id, frame)
        
        # Add to history
        self.orientation_history[track_id].append(detected_orientation)
        
        # Smooth orientation using majority vote
        if len(self.orientation_history[track_id]) >= 3:
            orientations = list(self.orientation_history[track_id])
            orientation_counts = {}
            for orient in orientations:
                orientation_counts[orient] = orientation_counts.get(orient, 0) + 1
            
            # Get most common orientation
            most_common = max(orientation_counts, key=orientation_counts.get)
            confidence = orientation_counts[most_common] / len(orientations)
            
            # Only update if confidence is high enough
            if confidence >= self.confidence_threshold and most_common != Orientation.UNKNOWN:
                previous_orientation = self.current_orientations[track_id]
                
                # Check for orientation change
                if most_common != previous_orientation and previous_orientation != Orientation.UNKNOWN:
                    change = OrientationChange(
                        timestamp=datetime.datetime.now(),
                        track_id=track_id,
                        previous_orientation=previous_orientation,
                        new_orientation=most_common,
                        confidence=confidence,
                        bounding_box=bbox
                    )
                    
                    self.orientation_changes.append(change)
                    self.current_orientations[track_id] = most_common
                    
                    return change
                
                # Update current orientation even if no change
                self.current_orientations[track_id] = most_common
        
        return None
    
    def get_current_orientation(self, track_id: int) -> Orientation:
        """Get current orientation for a tracked object"""
        return self.current_orientations.get(track_id, Orientation.UNKNOWN)
    
    def get_orientation_changes(self, since: Optional[datetime.datetime] = None) -> List[OrientationChange]:
        """Get orientation changes, optionally filtered by time"""
        if since is None:
            return self.orientation_changes.copy()
        
        return [change for change in self.orientation_changes if change.timestamp >= since]
    
    def draw_orientation_info(self, frame: np.ndarray, bbox: Tuple[float, float, float, float], 
                            track_id: int) -> np.ndarray:
        """Draw orientation information on frame"""
        orientation = self.get_current_orientation(track_id)
        
        if orientation != Orientation.UNKNOWN:
            x, y, w, h = bbox
            
            # Color coding for orientations
            color_map = {
                Orientation.UPRIGHT: (0, 255, 0),     # Green
                Orientation.FLAT: (0, 165, 255),      # Orange
                Orientation.UPSIDE_DOWN: (0, 0, 255), # Red
            }
            
            color = color_map.get(orientation, (128, 128, 128))
            
            # Draw orientation label
            label = f"Orient: {orientation.value}"
            cv2.putText(frame, label, 
                       (int(x - w/2), int(y + h/2) + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw colored border around object
            cv2.rectangle(frame, 
                         (int(x - w/2), int(y - h/2)), 
                         (int(x + w/2), int(y + h/2)), 
                         color, 2)
        
        return frame
    
    def cleanup_old_tracks(self, active_track_ids: List[int]):
        """Remove data for tracks that are no longer active"""
        current_tracks = set(self.orientation_history.keys())
        active_set = set(active_track_ids)
        
        for track_id in current_tracks - active_set:
            if track_id in self.orientation_history:
                del self.orientation_history[track_id]
            if track_id in self.current_orientations:
                del self.current_orientations[track_id]
            if track_id in self.previous_boxes:
                del self.previous_boxes[track_id]