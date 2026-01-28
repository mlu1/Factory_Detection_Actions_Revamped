import os
import json
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False
    return True

def setup_database():
    """Initialize the database"""
    print("🗄️ Setting up database...")
    try:
        from factory_monitor_enhanced import FactoryMonitoringSystem
        monitor = FactoryMonitoringSystem()
        print("✅ Database initialized successfully!")
    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        return False
    return True

def create_demo_data():
    """Create some demo data for testing"""
    print("📊 Creating demo data...")
    try:
        import sqlite3
        import datetime
        
        conn = sqlite3.connect("factory_monitoring.db")
        cursor = conn.cursor()
        
        # Create demo employee sessions
        base_time = datetime.datetime.now() - datetime.timedelta(hours=8)
        
        for i in range(3):
            employee_id = f"EMP_{i+1:03d}"
            
            # Create some item completions
            for j in range(10 + i*5):
                completion_time = base_time + datetime.timedelta(minutes=j*15)
                cursor.execute("""
                    INSERT INTO item_completions (timestamp, employee_id, item_type, completion_time, quality_score)
                    VALUES (?, ?, ?, ?, ?)
                """, (completion_time.isoformat(), employee_id, "Standard", 
                      45 + (i*10) + (j%20), 0.8 + (i*0.05)))
                
                # Create corresponding activity logs
                activities = ["Picking Up", "Processing", "Putting Down", "Handling"]
                for k, activity in enumerate(activities):
                    activity_time = completion_time + datetime.timedelta(seconds=k*10)
                    cursor.execute("""
                        INSERT INTO activity_logs (timestamp, employee_id, activity, duration)
                        VALUES (?, ?, ?, ?)
                    """, (activity_time.isoformat(), employee_id, activity, 10 + k*5))
        
        conn.commit()
        conn.close()
        print("✅ Demo data created successfully!")
        
    except Exception as e:
        print(f"❌ Error creating demo data: {e}")
        return False
    return True

def check_model_files():
    """Check if YOLO model files exist"""
    print("🔍 Checking YOLO model files...")
    models = ["yolo26x.pt", "yolov8n-pose.pt"]
    
    missing_models = []
    for model in models:
        if not os.path.exists(model):
            missing_models.append(model)
    
    if missing_models:
        print("⚠️  Missing model files:")
        for model in missing_models:
            print(f"   - {model}")
        print("\nNote: The pose model (yolov8n-pose.pt) will be downloaded automatically on first run.")
        print("Make sure you have your custom object detection model (yolo26x.pt) in the current directory.")
    else:
        print("✅ All model files found!")

def show_usage_instructions():
    """Show usage instructions"""
    print("\n" + "="*60)
    print("🏭 FACTORY MONITORING SYSTEM SETUP COMPLETE!")
    print("="*60)
    print("\n📋 HOW TO USE:")
    print("\n1. Main Monitoring Application:")
    print("   python factory_monitor_enhanced.py")
    print("\n2. Analytics Dashboard:")
    print("   python analytics_dashboard.py")
    print("\n3. Web Dashboard (run in separate terminal):")
    print("   python web_dashboard.py")
    print("   Then open: http://localhost:5000")
    print("\n⌨️  KEYBOARD SHORTCUTS (during monitoring):")
    print("   'q' - Quit application")
    print("   'r' - Generate report")
    print("   's' - Print current summary")
    print("\n📁 OUTPUT FILES:")
    print("   - factory_monitoring.db (SQLite database)")
    print("   - factory_report_YYYY-MM-DD.json (daily reports)")
    print("   - Various chart images (PNG files)")
    print("\n🔧 CONFIGURATION:")
    print("   Edit config.json to customize thresholds and settings")
    print("\n" + "="*60)

def main():
    print("🚀 Starting Factory Monitoring System Setup")
    print("="*50)
    
    success = True
    
    # Install requirements
    if not install_requirements():
        success = False
    
    # Setup database
    if success and not setup_database():
        success = False
    
    # Create demo data
    if success and not create_demo_data():
        success = False
    
    # Check model files
    check_model_files()
    
    if success:
        show_usage_instructions()
    else:
        print("\n❌ Setup completed with errors. Please check the error messages above.")

if __name__ == "__main__":
    main()