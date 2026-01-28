# 🏭 Factory Monitoring System

A comprehensive AI-powered factory monitoring system that tracks employee activities, counts completed items, and provides real-time analytics using computer vision.

## ✨ Features

### 🎯 Core Functionality
- **Employee Detection & Tracking**: Automatic detection and tracking of workers using YOLO
- **Activity Recognition**: Identifies activities like picking up, putting down, processing, carrying, etc.
- **Item Completion Counting**: Automatically counts items completed by each worker
- **Real-time Monitoring**: Live video analysis with overlay information

### 📊 Analytics & Reporting
- **Productivity Metrics**: Items per hour, completion times, efficiency scores
- **Performance Analytics**: Employee comparison, trend analysis
- **Visual Reports**: Charts, graphs, and heatmaps
- **Export Capabilities**: Excel reports, JSON summaries, PDF charts

### 🌐 Dashboard & Interface
- **Web Dashboard**: Real-time monitoring via web browser
- **Live Statistics**: Current production rates, active employees
- **Historical Data**: Performance trends and historical analysis
- **Alert System**: Configurable productivity and quality alerts

## 📁 System Architecture

```
FactoryDetectionActions/
├── factory_monitor_enhanced.py    # Main monitoring application
├── analytics_dashboard.py         # Analytics and reporting system
├── web_dashboard.py               # Flask web interface
├── config.json                    # Configuration settings
├── setup.py                       # Setup and installation script
├── requirements.txt               # Python dependencies
├── templates/
│   └── dashboard.html            # Web dashboard template
├── yolo26x.pt                    # Custom object detection model
├── yolov8n-pose.pt              # Pose estimation model
└── factory_monitoring.db         # SQLite database (created on run)
```

## 🚀 Quick Start

### 1. Installation
```bash
# Run the setup script to install dependencies and initialize the system
python setup.py
```

### 2. Basic Usage
```bash
# Start the main monitoring application
python factory_monitor_enhanced.py

# Generate analytics reports (separate terminal)
python analytics_dashboard.py

# Start web dashboard (separate terminal)
python web_dashboard.py
# Then open: http://localhost:5000
```

## 🎮 Controls & Shortcuts

### During Video Monitoring
- **`q`** - Quit application
- **`r`** - Generate and export daily report
- **`s`** - Print current shift summary to console

### Configuration
Edit `config.json` to customize:
- Detection thresholds
- Activity definitions
- Alert settings
- Video sources
- Database settings

## 📋 Employee Activity Types

The system recognizes these activities:

| Activity | Description | Completion Trigger |
|----------|-------------|-------------------|
| **Picking Up** | Worker lifting/grabbing an item | ✓ |
| **Putting Down** | Worker placing item down | ✓ |
| **Processing** | Working on item (hand moving, object stationary) | ✓ |
| **Carrying** | Moving item horizontally | - |
| **Handling** | General object manipulation | - |
| **Holding** | Stationary grip on object | - |
| **Touching** | Brief contact with object | - |

## 📊 Analytics Features

### Real-time Metrics
- Total items produced today
- Currently active employees  
- Current hour production rate
- Average completion time

### Performance Analytics
- **Employee Comparison**: Side-by-side productivity metrics
- **Hourly Trends**: Production patterns throughout the day
- **Activity Breakdown**: Time spent on each activity type
- **Quality Scores**: Item completion quality assessment

### Export Options
- **Excel Reports**: Comprehensive daily/weekly summaries
- **JSON Data**: Raw data for custom analysis
- **Chart Images**: Production graphs and performance charts
- **Database Backup**: Full SQLite database export

## 🔧 Configuration Options

### Detection Settings
```json
{
  "detection_config": {
    "confidence_threshold": 0.5,
    "tracking_buffer_frames": 30,
    "smoothing_frames": 10
  }
}
```

### Productivity Settings
```json
{
  "monitoring_config": {
    "min_handling_time_seconds": 2.0,
    "productivity_target_items_per_hour": 10,
    "completion_activities": ["Putting Down", "Processing"]
  }
}
```

### Alert Configuration
```json
{
  "alerts": {
    "low_productivity_threshold": 50,
    "extended_idle_time_minutes": 10,
    "enable_email_alerts": false
  }
}
```

## 🎥 Video Sources

The system supports multiple video input sources:

### Local Video Files
```python
video_source = "test_video.mp4"
```

### RTSP Camera Streams
```python
rtsp_url = "rtsp://admin:password@192.168.1.100:554/stream1"
```

### USB/Webcam
```python
camera_id = 0  # Default camera
```

## 📈 Database Schema

The system uses SQLite with these main tables:

### Employee Sessions
- Session tracking and productivity metrics
- Start/end times and total statistics

### Item Completions  
- Individual item completion records
- Timestamps, employee ID, completion time, quality score

### Activity Logs
- Detailed activity tracking
- Duration and type of each activity

## 🔍 Troubleshooting

### Common Issues

**Model Files Missing**
```bash
# The pose model downloads automatically
# Ensure your custom model (yolo26x.pt) is in the project directory
```

**Database Errors**
```bash
# Reinitialize database
python -c "from factory_monitor_enhanced import FactoryMonitoringSystem; FactoryMonitoringSystem()"
```

**Performance Issues**
- Reduce video resolution in config.json
- Adjust detection confidence thresholds
- Use smaller YOLO model variants

**Accuracy Issues**
- Calibrate interaction thresholds in config.json
- Adjust smoothing parameters
- Retrain custom object detection model

## 📊 Sample Analytics Output

### Daily Summary Example
```json
{
  "date": "2026-01-28",
  "total_items_produced": 247,
  "total_employees": 8,
  "shift_duration": 8.5,
  "employees": [
    {
      "employee_id": "EMP_001",
      "items_completed": 35,
      "productivity_score": 87.3,
      "current_activity": "Processing"
    }
  ]
}
```

### Performance Metrics
- **Items per Hour**: Average production rate per employee
- **Completion Time**: Average time to complete each item
- **Activity Distribution**: Percentage time spent on each activity
- **Quality Score**: Item completion quality assessment
- **Productivity Trends**: Performance over time

## 🤝 Contributing

This system is designed to be extensible. Key areas for enhancement:

### Activity Recognition
- Add new activity types
- Improve detection accuracy
- Integrate additional sensors

### Analytics
- Advanced ML models for predictions
- Quality assessment using computer vision
- Predictive maintenance alerts

### Integration
- ERP system connectivity
- IoT sensor integration
- Mobile app development

## 📄 License

This project is for factory monitoring and productivity analysis purposes. Ensure compliance with workplace monitoring regulations in your jurisdiction.

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section
2. Review configuration settings
3. Examine database logs
4. Test with demo data

## 🔄 Updates & Maintenance

### Regular Tasks
- **Daily**: Review production reports
- **Weekly**: Analyze employee performance trends  
- **Monthly**: Update detection models and thresholds
- **Quarterly**: Database cleanup and optimization

### Model Updates
- Retrain object detection models with new data
- Update pose estimation for different worker positions
- Calibrate thresholds based on actual performance data

---

🏭 **Happy Monitoring!** This system provides comprehensive insights into your factory operations while respecting worker privacy and safety.