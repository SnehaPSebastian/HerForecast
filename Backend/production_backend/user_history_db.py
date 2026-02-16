"""
User History Storage with SQLite
Persistent storage for historical data tracking
"""

import sqlite3
import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path

class UserHistoryDB:
    """
    SQLite-based persistent storage for user history
    Stores last 21 days per user for rolling window calculations
    """
    
    def __init__(self, db_path: str = 'user_history.db'):
        """Initialize database"""
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Create tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                date TEXT NOT NULL,
                rmssd_mean REAL,
                wrist_temp_mean REAL,
                estrogen REAL,
                pdg REAL,
                lh REAL,
                stress_score_mean REAL,
                oxygen_ratio_mean REAL,
                day_in_study REAL,
                predicted_phase TEXT,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, date)
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_user_date 
            ON user_data(user_id, date DESC)
        ''')
        
        conn.commit()
        conn.close()
    
    def add_entry(self, user_id: str, date: str, data: Dict):
        """
        Add or update a daily entry
        
        Args:
            user_id: User identifier
            date: Date in YYYY-MM-DD format
            data: Dictionary with all features
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO user_data (
                user_id, date, rmssd_mean, wrist_temp_mean, estrogen, pdg, lh,
                stress_score_mean, oxygen_ratio_mean, day_in_study,
                predicted_phase, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id, date,
            data.get('rmssd_mean'),
            data.get('wrist_temp_mean'),
            data.get('estrogen'),
            data.get('pdg'),
            data.get('lh'),
            data.get('stress_score_mean'),
            data.get('oxygen_ratio_mean'),
            data.get('day_in_study'),
            data.get('predicted_phase'),
            data.get('confidence')
        ))
        
        conn.commit()
        conn.close()
        
        # Clean old entries (keep last 30 days)
        self._cleanup_old_entries(user_id, days=30)
    
    def get_history(self, user_id: str, days: int = 21) -> List[Dict]:
        """
        Get user's recent history
        
        Args:
            user_id: User identifier
            days: Number of days to retrieve
            
        Returns:
            List of dictionaries with historical data
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM user_data
            WHERE user_id = ?
            ORDER BY date DESC
            LIMIT ?
        ''', (user_id, days))
        
        rows = cursor.fetchall()
        conn.close()
        
        # Convert to list of dicts (reverse to chronological order)
        history = [dict(row) for row in reversed(rows)]
        
        return history
    
    def has_sufficient_history(self, user_id: str, min_days: int = 7) -> bool:
        """Check if user has enough history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) FROM user_data
            WHERE user_id = ?
        ''', (user_id,))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count >= min_days
    
    def get_cycle_stats(self, user_id: str) -> Dict:
        """
        Calculate cycle statistics
        
        Returns:
            Dictionary with cycle metrics
        """
        history = self.get_history(user_id, days=30)
        
        if len(history) < 7:
            return {}
        
        stats = {}
        
        # Find menstrual phases
        menstrual_days = [
            i for i, entry in enumerate(history)
            if entry.get('predicted_phase') == 'Menstrual'
        ]
        
        if menstrual_days:
            stats['days_since_menstruation'] = len(history) - menstrual_days[-1] - 1
            stats['last_menstrual_date'] = history[menstrual_days[-1]]['date']
        
        # Cycle regularity
        if len(menstrual_days) >= 2:
            import numpy as np
            gaps = np.diff(menstrual_days)
            avg_cycle = np.mean(gaps)
            std_cycle = np.std(gaps)
            
            stats['average_cycle_length'] = float(avg_cycle)
            stats['cycle_std'] = float(std_cycle)
            stats['is_regular'] = std_cycle < 3  # Regular if std < 3 days
        
        # Hormone trends
        recent_estrogen = [e.get('estrogen', 0) for e in history[-7:] if e.get('estrogen') is not None]
        if recent_estrogen:
            import numpy as np
            stats['estrogen_trend'] = 'rising' if np.polyfit(range(len(recent_estrogen)), recent_estrogen, 1)[0] > 0 else 'falling'
        
        return stats
    
    def _cleanup_old_entries(self, user_id: str, days: int = 30):
        """Remove entries older than specified days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM user_data
            WHERE user_id = ?
            AND date < date('now', '-' || ? || ' days')
        ''', (user_id, days))
        
        conn.commit()
        conn.close()
    
    def export_user_data(self, user_id: str, output_file: str):
        """Export user data to JSON"""
        history = self.get_history(user_id, days=90)
        
        with open(output_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def get_all_users(self) -> List[str]:
        """Get list of all user IDs"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT DISTINCT user_id FROM user_data')
        users = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return users


# Example usage
if __name__ == "__main__":
    # Initialize database
    db = UserHistoryDB('user_history.db')
    
    # Add sample data
    db.add_entry(
        user_id='test_user_001',
        date='2024-01-01',
        data={
            'rmssd_mean': 0.2,
            'wrist_temp_mean': 0.1,
            'estrogen': 0.3,
            'pdg': -0.1,
            'lh': 0.5,
            'stress_score_mean': -0.05,
            'oxygen_ratio_mean': 0.0,
            'day_in_study': 0.1,
            'predicted_phase': 'Follicular',
            'confidence': 0.65
        }
    )
    
    # Get history
    history = db.get_history('test_user_001', days=7)
    print(f"History entries: {len(history)}")
    
    # Get cycle stats
    stats = db.get_cycle_stats('test_user_001')
    print(f"Cycle stats: {stats}")
    
    print("\n✅ User history database working!")
