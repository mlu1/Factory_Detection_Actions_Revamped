import sqlite3
import json
import datetime
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

class FactoryAnalyticsDashboard:
    """Analytics and reporting dashboard for factory monitoring data"""
    
    def __init__(self, db_path="factory_monitoring.db"):
        self.db_path = db_path
        
    def get_employee_performance(self, employee_id: str = None, date_range: tuple = None) -> pd.DataFrame:
        """Get employee performance metrics"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT 
            ic.employee_id,
            DATE(ic.timestamp) as date,
            COUNT(*) as items_completed,
            AVG(ic.completion_time) as avg_completion_time,
            SUM(ic.completion_time) as total_work_time,
            AVG(ic.quality_score) as avg_quality_score
        FROM item_completions ic
        WHERE 1=1
        """
        
        params = []
        if employee_id:
            query += " AND ic.employee_id = ?"
            params.append(employee_id)
            
        if date_range:
            query += " AND DATE(ic.timestamp) BETWEEN ? AND ?"
            params.extend(date_range)
            
        query += " GROUP BY ic.employee_id, DATE(ic.timestamp) ORDER BY DATE(ic.timestamp)"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df
        
    def get_hourly_production(self, date: str = None) -> pd.DataFrame:
        """Get hourly production statistics"""
        conn = sqlite3.connect(self.db_path)
        
        if date is None:
            date = datetime.datetime.now().strftime("%Y-%m-%d")
            
        query = """
        SELECT 
            strftime('%H', timestamp) as hour,
            COUNT(*) as items_produced,
            COUNT(DISTINCT employee_id) as active_employees,
            AVG(completion_time) as avg_completion_time
        FROM item_completions 
        WHERE DATE(timestamp) = ?
        GROUP BY strftime('%H', timestamp)
        ORDER BY hour
        """
        
        df = pd.read_sql_query(query, conn, params=[date])
        conn.close()
        return df
        
    def get_activity_breakdown(self, employee_id: str = None) -> pd.DataFrame:
        """Get breakdown of activities by time spent"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT 
            employee_id,
            activity,
            COUNT(*) as activity_count,
            SUM(duration) as total_time,
            AVG(duration) as avg_duration
        FROM activity_logs
        WHERE 1=1
        """
        
        params = []
        if employee_id:
            query += " AND employee_id = ?"
            params.append(employee_id)
            
        query += " GROUP BY employee_id, activity ORDER BY total_time DESC"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df
        
    def generate_production_chart(self, date: str = None, save_path: str = "production_chart.png"):
        """Generate hourly production chart"""
        df = self.get_hourly_production(date)
        
        if df.empty:
            print("No data available for production chart")
            return None
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Items produced per hour
        ax1.bar(df['hour'].astype(int), df['items_produced'], color='skyblue', alpha=0.7)
        ax1.set_title(f'Hourly Production - {date or "Today"}')
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Items Produced')
        ax1.grid(True, alpha=0.3)
        
        # Active employees per hour
        ax2.plot(df['hour'].astype(int), df['active_employees'], marker='o', color='green', linewidth=2)
        ax2.set_title('Active Employees by Hour')
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Number of Active Employees')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
        
    def generate_employee_performance_chart(self, days: int = 7, save_path: str = "employee_performance.png"):
        """Generate employee performance comparison chart"""
        end_date = datetime.datetime.now().date()
        start_date = end_date - datetime.timedelta(days=days)
        
        df = self.get_employee_performance(date_range=(str(start_date), str(end_date)))
        
        if df.empty:
            print("No employee performance data available")
            return None
            
        # Create summary by employee
        employee_summary = df.groupby('employee_id').agg({
            'items_completed': 'sum',
            'avg_completion_time': 'mean',
            'avg_quality_score': 'mean'
        }).reset_index()
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Items completed
        ax1.bar(employee_summary['employee_id'], employee_summary['items_completed'])
        ax1.set_title('Total Items Completed')
        ax1.set_xlabel('Employee ID')
        ax1.set_ylabel('Items Completed')
        ax1.tick_params(axis='x', rotation=45)
        
        # Average completion time
        ax2.bar(employee_summary['employee_id'], employee_summary['avg_completion_time'], color='orange')
        ax2.set_title('Average Completion Time')
        ax2.set_xlabel('Employee ID')
        ax2.set_ylabel('Time (seconds)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Quality scores
        ax3.bar(employee_summary['employee_id'], employee_summary['avg_quality_score'], color='green')
        ax3.set_title('Average Quality Score')
        ax3.set_xlabel('Employee ID')
        ax3.set_ylabel('Quality Score')
        ax3.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
        
    def generate_activity_heatmap(self, save_path: str = "activity_heatmap.png"):
        """Generate activity heatmap by employee and hour"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT 
            employee_id,
            strftime('%H', timestamp) as hour,
            activity,
            SUM(duration) as total_time
        FROM activity_logs 
        WHERE DATE(timestamp) = DATE('now')
        GROUP BY employee_id, strftime('%H', timestamp), activity
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            print("No activity data available for heatmap")
            return None
            
        # Pivot for heatmap
        heatmap_data = df.pivot_table(
            values='total_time', 
            index='employee_id', 
            columns='hour', 
            aggfunc='sum', 
            fill_value=0
        )
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='.0f')
        plt.title('Employee Activity Heatmap (Total Time by Hour)')
        plt.xlabel('Hour of Day')
        plt.ylabel('Employee ID')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
        
    def export_daily_summary_excel(self, date: str = None, filename: str = None):
        """Export comprehensive daily summary to Excel"""
        if date is None:
            date = datetime.datetime.now().strftime("%Y-%m-%d")
            
        if filename is None:
            filename = f"factory_summary_{date}.xlsx"
            
        # Get all data
        performance_df = self.get_employee_performance(date_range=(date, date))
        hourly_df = self.get_hourly_production(date)
        activity_df = self.get_activity_breakdown()
        
        # Write to Excel with multiple sheets
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            performance_df.to_excel(writer, sheet_name='Employee Performance', index=False)
            hourly_df.to_excel(writer, sheet_name='Hourly Production', index=False)
            activity_df.to_excel(writer, sheet_name='Activity Breakdown', index=False)
            
            # Create summary sheet
            summary_data = {
                'Metric': [
                    'Total Items Produced',
                    'Total Active Employees',
                    'Average Completion Time',
                    'Peak Production Hour',
                    'Total Production Time'
                ],
                'Value': [
                    performance_df['items_completed'].sum() if not performance_df.empty else 0,
                    performance_df['employee_id'].nunique() if not performance_df.empty else 0,
                    performance_df['avg_completion_time'].mean() if not performance_df.empty else 0,
                    hourly_df.loc[hourly_df['items_produced'].idxmax(), 'hour'] if not hourly_df.empty else 'N/A',
                    activity_df['total_time'].sum() if not activity_df.empty else 0
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Daily Summary', index=False)
            
        return filename
        
    def generate_productivity_report(self) -> Dict:
        """Generate comprehensive productivity report"""
        conn = sqlite3.connect(self.db_path)
        
        # Overall metrics
        total_items_query = "SELECT COUNT(*) as total_items FROM item_completions WHERE DATE(timestamp) = DATE('now')"
        total_items = pd.read_sql_query(total_items_query, conn).iloc[0]['total_items']
        
        # Employee efficiency
        efficiency_query = """
        SELECT 
            employee_id,
            COUNT(*) as items_completed,
            SUM(completion_time) as total_time,
            AVG(completion_time) as avg_time,
            (COUNT(*) * 1.0 / (SUM(completion_time) / 3600)) as items_per_hour
        FROM item_completions 
        WHERE DATE(timestamp) = DATE('now')
        GROUP BY employee_id
        ORDER BY items_per_hour DESC
        """
        efficiency_df = pd.read_sql_query(efficiency_query, conn)
        
        # Peak hours
        peak_query = """
        SELECT 
            strftime('%H', timestamp) as hour,
            COUNT(*) as items_produced
        FROM item_completions 
        WHERE DATE(timestamp) = DATE('now')
        GROUP BY strftime('%H', timestamp)
        ORDER BY items_produced DESC
        LIMIT 3
        """
        peak_df = pd.read_sql_query(peak_query, conn)
        
        conn.close()
        
        report = {
            'date': datetime.datetime.now().strftime("%Y-%m-%d"),
            'total_items_today': int(total_items),
            'top_performers': efficiency_df.head(3).to_dict('records') if not efficiency_df.empty else [],
            'peak_production_hours': peak_df.to_dict('records') if not peak_df.empty else [],
            'overall_efficiency': {
                'avg_items_per_hour': efficiency_df['items_per_hour'].mean() if not efficiency_df.empty else 0,
                'total_active_employees': len(efficiency_df) if not efficiency_df.empty else 0
            }
        }
        
        return report

def main():
    """Demo of analytics dashboard"""
    dashboard = FactoryAnalyticsDashboard()
    
    # Generate reports
    print("Generating factory analytics reports...")
    
    try:
        # Production chart
        chart_path = dashboard.generate_production_chart()
        if chart_path:
            print(f"✅ Production chart saved: {chart_path}")
            
        # Employee performance chart
        performance_path = dashboard.generate_employee_performance_chart()
        if performance_path:
            print(f"✅ Employee performance chart saved: {performance_path}")
            
        # Activity heatmap
        heatmap_path = dashboard.generate_activity_heatmap()
        if heatmap_path:
            print(f"✅ Activity heatmap saved: {heatmap_path}")
            
        # Excel summary
        excel_path = dashboard.export_daily_summary_excel()
        print(f"✅ Daily summary exported: {excel_path}")
        
        # Productivity report
        report = dashboard.generate_productivity_report()
        print("\n📊 PRODUCTIVITY REPORT")
        print("=" * 50)
        print(json.dumps(report, indent=2))
        
    except Exception as e:
        print(f"❌ Error generating reports: {e}")
        print("Make sure you have data in the database and required packages installed:")
        print("pip install matplotlib pandas seaborn openpyxl")

if __name__ == "__main__":
    main()