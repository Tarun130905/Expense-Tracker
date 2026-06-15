"""
Initialize database schema and sample data.
Run this script once to set up the database.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import ExpenseDatabase


def init_database():
    """Initialize database with schema."""
    db = ExpenseDatabase()
    db.init_schema()
    print("✓ Database schema initialized successfully")
    db.close()


def load_sample_data():
    """Load sample transactions from CSV."""
    db = ExpenseDatabase()
    
    sample_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'sample_transactions.csv'
    )
    
    if os.path.exists(sample_file):
        count = db.bulk_insert_from_csv(sample_file)
        print(f"✓ Loaded {count} sample transactions")
    else:
        print(f"⚠ Sample file not found: {sample_file}")
    
    db.close()


if __name__ == '__main__':
    print("Initializing expense tracker database...")
    init_database()
    
    print("\nLoading sample data...")
    load_sample_data()
    
    print("\n✓ Database setup complete!")
