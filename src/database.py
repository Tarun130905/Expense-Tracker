"""
Database module for SQLite expense tracking.
Handles CRUD operations, schema initialization, and data persistence.
"""

import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import pandas as pd


class ExpenseDatabase:
    """SQLite database wrapper for expense tracking."""
    
    def __init__(self, db_path: str = "data/expenses.db"):
        """Initialize database connection."""
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = None
        self.connect()
    
    def connect(self) -> None:
        """Establish database connection."""
        # check_same_thread=False allows SQLite to be used in multi-threaded environments (Streamlit)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
    
    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def init_schema(self) -> None:
        """Create database schema if not exists."""
        cursor = self.conn.cursor()
        
        # Create transactions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            amount REAL NOT NULL,
            vendor TEXT,
            description TEXT,
            category TEXT,
            confidence REAL DEFAULT 0.0,
            is_anomaly INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create categories table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            color_code TEXT
        )
        ''')
        
        # Insert default categories
        default_categories = [
            ('Groceries', 'Food and groceries', '#1f77b4'),
            ('Transportation', 'Fuel, transit, rideshare', '#ff7f0e'),
            ('Utilities', 'Electric, water, internet', '#2ca02c'),
            ('Entertainment', 'Movies, games, subscriptions', '#d62728'),
            ('Dining', 'Restaurants and takeout', '#9467bd'),
            ('Healthcare', 'Medical and pharmacy', '#8c564b'),
            ('Rent', 'Housing', '#e377c2'),
            ('Misc', 'Other expenses', '#7f7f7f')
        ]
        
        for cat_name, desc, color in default_categories:
            try:
                cursor.execute(
                    'INSERT INTO categories (name, description, color_code) VALUES (?, ?, ?)',
                    (cat_name, desc, color)
                )
            except sqlite3.IntegrityError:
                pass  # Category already exists
        
        self.conn.commit()
    
    def add_transaction(self, date: str, amount: float, vendor: str, 
                       description: str, category: str = None, 
                       confidence: float = 0.0) -> int:
        """Add a new transaction to database."""
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO transactions 
        (date, amount, vendor, description, category, confidence)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (date, amount, vendor, description, category, confidence))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_transactions(self, limit: int = 100) -> pd.DataFrame:
        """Retrieve all transactions as DataFrame."""
        query = '''
        SELECT * FROM transactions 
        ORDER BY date DESC 
        LIMIT ?
        '''
        return pd.read_sql_query(query, self.conn, params=(limit,))
    
    def get_transactions_by_date_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get transactions within date range."""
        query = '''
        SELECT * FROM transactions 
        WHERE date BETWEEN ? AND ?
        ORDER BY date DESC
        '''
        return pd.read_sql_query(query, self.conn, params=(start_date, end_date))
    
    def get_transactions_by_category(self, category: str) -> pd.DataFrame:
        """Get transactions for specific category."""
        query = '''
        SELECT * FROM transactions 
        WHERE category = ?
        ORDER BY date DESC
        '''
        return pd.read_sql_query(query, self.conn, params=(category,))
    
    def update_transaction(self, transaction_id: int, category: str = None, 
                          is_anomaly: int = None) -> None:
        """Update transaction fields."""
        cursor = self.conn.cursor()
        updates = []
        params = []
        
        if category is not None:
            updates.append("category = ?")
            params.append(category)
        if is_anomaly is not None:
            updates.append("is_anomaly = ?")
            params.append(is_anomaly)
        
        if updates:
            updates.append("updated_at = CURRENT_TIMESTAMP")
            params.append(transaction_id)
            query = f"UPDATE transactions SET {', '.join(updates)} WHERE id = ?"
            cursor.execute(query, params)
            self.conn.commit()
    
    def delete_transaction(self, transaction_id: int) -> None:
        """Delete a transaction."""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM transactions WHERE id = ?', (transaction_id,))
        self.conn.commit()
    
    def get_categories(self) -> List[str]:
        """Get list of all categories."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT name FROM categories ORDER BY name')
        return [row[0] for row in cursor.fetchall()]
    
    def get_category_stats(self, category: str = None) -> pd.DataFrame:
        """Get spending statistics by category."""
        if category:
            query = '''
            SELECT category, 
                   COUNT(*) as transaction_count,
                   SUM(amount) as total_spending,
                   AVG(amount) as avg_amount,
                   MIN(amount) as min_amount,
                   MAX(amount) as max_amount
            FROM transactions
            WHERE category = ?
            GROUP BY category
            '''
            return pd.read_sql_query(query, self.conn, params=(category,))
        else:
            query = '''
            SELECT category, 
                   COUNT(*) as transaction_count,
                   SUM(amount) as total_spending,
                   AVG(amount) as avg_amount,
                   MIN(amount) as min_amount,
                   MAX(amount) as max_amount
            FROM transactions
            GROUP BY category
            ORDER BY total_spending DESC
            '''
            return pd.read_sql_query(query, self.conn)
    
    def get_monthly_stats(self) -> pd.DataFrame:
        """Get monthly spending totals."""
        query = '''
        SELECT 
            substr(date, 1, 7) as month,
            COUNT(*) as transaction_count,
            SUM(amount) as total_spending,
            AVG(amount) as avg_amount
        FROM transactions
        GROUP BY substr(date, 1, 7)
        ORDER BY month DESC
        '''
        return pd.read_sql_query(query, self.conn)
    
    def get_monthly_by_category(self) -> pd.DataFrame:
        """Get monthly spending by category."""
        query = '''
        SELECT 
            substr(date, 1, 7) as month,
            category,
            SUM(amount) as amount
        FROM transactions
        WHERE category IS NOT NULL
        GROUP BY month, category
        ORDER BY month DESC
        '''
        return pd.read_sql_query(query, self.conn)
    
    def bulk_insert_from_csv(self, csv_path: str) -> int:
        """Import transactions from CSV file."""
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        required_cols = ['date', 'amount']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")
        
        cursor = self.conn.cursor()
        count = 0
        
        for _, row in df.iterrows():
            try:
                vendor = row.get('vendor', '')
                description = row.get('description', '')
                category = row.get('category', None)
                
                cursor.execute('''
                INSERT INTO transactions 
                (date, amount, vendor, description, category)
                VALUES (?, ?, ?, ?, ?)
                ''', (row['date'], row['amount'], vendor, description, category))
                count += 1
            except Exception as e:
                print(f"Error inserting row: {e}")
                continue
        
        self.conn.commit()
        return count
    
    def export_to_csv(self, output_path: str, start_date: str = None, 
                      end_date: str = None) -> None:
        """Export transactions to CSV."""
        if start_date and end_date:
            df = self.get_transactions_by_date_range(start_date, end_date)
        else:
            df = self.get_transactions(limit=10000)
        
        df.to_csv(output_path, index=False)
    
    def get_anomalies(self) -> pd.DataFrame:
        """Get transactions flagged as anomalies."""
        query = '''
        SELECT * FROM transactions 
        WHERE is_anomaly = 1
        ORDER BY date DESC
        '''
        return pd.read_sql_query(query, self.conn)
    
    def get_total_spending(self, start_date: str = None, end_date: str = None) -> float:
        """Get total spending in date range."""
        if start_date and end_date:
            query = 'SELECT SUM(amount) FROM transactions WHERE date BETWEEN ? AND ?'
            cursor = self.conn.cursor()
            cursor.execute(query, (start_date, end_date))
        else:
            query = 'SELECT SUM(amount) FROM transactions'
            cursor = self.conn.cursor()
            cursor.execute(query)
        
        result = cursor.fetchone()[0]
        return result if result else 0.0
