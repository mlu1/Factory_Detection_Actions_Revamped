from flask import Flask, render_template, jsonify, request
import sqlite3
import json
import datetime
from analytics_dashboard import FactoryAnalyticsDashboard

app = Flask(__name__)
dashboard = FactoryAnalyticsDashboard()

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/live-stats')
def live_stats():
    """Get current live statistics"""
    conn = sqlite3.connect('factory_monitoring.db')
    cursor = conn.cursor()
    
    # Get today's stats
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # Total items today
    cursor.execute("""
        SELECT COUNT(*) FROM item_completions 
        WHERE DATE(timestamp) = ?
    """, (today,))
    total_items = cursor.fetchone()[0]
    
    # Active employees (employees who completed items in last hour)
    one_hour_ago = (datetime.datetime.now() - datetime.timedelta(hours=1)).isoformat()
    cursor.execute("""
        SELECT COUNT(DISTINCT employee_id) FROM item_completions 
        WHERE timestamp > ?
    """, (one_hour_ago,))
    active_employees = cursor.fetchone()[0]
    
    # Current hour production
    current_hour = datetime.datetime.now().strftime('%Y-%m-%d %H')
    cursor.execute("""
        SELECT COUNT(*) FROM item_completions 
        WHERE strftime('%Y-%m-%d %H', timestamp) = ?
    """, (current_hour,))
    current_hour_items = cursor.fetchone()[0]
    
    # Average completion time today
    cursor.execute("""
        SELECT AVG(completion_time) FROM item_completions 
        WHERE DATE(timestamp) = ?
    """, (today,))
    avg_time = cursor.fetchone()[0] or 0
    
    conn.close()
    
    return jsonify({
        'total_items_today': total_items,
        'active_employees': active_employees,
        'current_hour_production': current_hour_items,
        'avg_completion_time': round(avg_time, 2),
        'timestamp': datetime.datetime.now().isoformat()
    })

@app.route('/api/employee-performance')
def employee_performance():
    """Get employee performance data"""
    days = request.args.get('days', 1, type=int)
    end_date = datetime.datetime.now().date()
    start_date = end_date - datetime.timedelta(days=days)
    
    df = dashboard.get_employee_performance(date_range=(str(start_date), str(end_date)))
    
    # Convert to JSON-serializable format
    if not df.empty:
        performance_data = df.groupby('employee_id').agg({
            'items_completed': 'sum',
            'avg_completion_time': 'mean',
            'avg_quality_score': 'mean'
        }).reset_index().to_dict('records')
    else:
        performance_data = []
    
    return jsonify(performance_data)

@app.route('/api/hourly-production')
def hourly_production():
    """Get hourly production data"""
    date = request.args.get('date', datetime.datetime.now().strftime('%Y-%m-%d'))
    df = dashboard.get_hourly_production(date)
    
    if not df.empty:
        hourly_data = df.to_dict('records')
        # Convert hour to int for better charting
        for item in hourly_data:
            item['hour'] = int(item['hour'])
    else:
        hourly_data = []
    
    return jsonify(hourly_data)

@app.route('/api/activity-summary')
def activity_summary():
    """Get activity breakdown summary"""
    df = dashboard.get_activity_breakdown()
    
    if not df.empty:
        activity_data = df.to_dict('records')
    else:
        activity_data = []
    
    return jsonify(activity_data)

@app.route('/api/productivity-report')
def productivity_report():
    """Get comprehensive productivity report"""
    report = dashboard.generate_productivity_report()
    return jsonify(report)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)