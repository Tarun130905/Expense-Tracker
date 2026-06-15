# Setup Guide - AI-Enhanced Expense Tracker

Complete step-by-step instructions for setting up the AI-Enhanced Expense Tracker on your local machine and deploying to Streamlit Cloud.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [Database Initialization](#database-initialization)
4. [Running the Application](#running-the-application)
5. [Training Models](#training-models)
6. [Deployment to Streamlit Cloud](#deployment-to-streamlit-cloud)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before starting, ensure you have:

- **Python 3.9 or higher** installed
  - Check: `python --version`
  - Download from: https://www.python.org/downloads/
  
- **Git** (optional, for version control)
  - Check: `git --version`
  - Download from: https://git-scm.com/

- **PowerShell 5.1+** (Windows, or Command Prompt)
  
- **4GB RAM** minimum (2GB recommended for dependencies)

- **Internet connection** for dependency downloads

---

## Local Development Setup

### Step 1: Clone or Create Project Directory

If using Git:
```powershell
git clone https://github.com/yourusername/expense-tracker.git
cd expense-tracker
```

Or manually create and navigate:
```powershell
mkdir expense-tracker
cd expense-tracker
```

### Step 2: Create Virtual Environment

Creating a virtual environment isolates project dependencies from system Python.

**On Windows (PowerShell):**
```powershell
python -m venv .venv
```

**Verify virtual environment creation:**
```powershell
ls -Path .venv
```

### Step 3: Activate Virtual Environment

**Windows PowerShell:**
```powershell
.venv\Scripts\Activate.ps1
```

If you get an execution policy error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then retry activation. You should see `(.venv)` prefix in your terminal.

**Verify activation:**
```powershell
python -c "import sys; print(sys.executable)"
```
Output should include `.venv\Scripts\python.exe`

### Step 4: Upgrade pip

Ensure you have the latest pip version:
```powershell
python -m pip install --upgrade pip
```

### Step 5: Create requirements.txt

Create a file named `requirements.txt` in the project root with the following contents:

```
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
xgboost==2.0.0
plotly==5.17.0
python-dotenv==1.0.0
sqlalchemy==2.0.20
tqdm==4.66.0
```

### Step 6: Install Dependencies

```powershell
pip install -r requirements.txt
```

This will take 2-5 minutes depending on internet speed. Wait for completion.

**Verify installation:**
```powershell
pip list
```

You should see all packages listed. Specifically check:
- streamlit ✓
- xgboost ✓
- scikit-learn ✓
- pandas ✓

---

## Database Initialization

### Step 1: Create Database Schema

Create `scripts/init_db.py`:

```python
import sqlite3
import os

def init_database(db_path='data/expenses.db'):
    """Initialize SQLite database with required tables."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
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
    
    conn.commit()
    conn.close()
    print("✓ Database initialized successfully")

if __name__ == '__main__':
    init_database()
```

### Step 2: Run Initialization Script

```powershell
python scripts/init_db.py
```

**Verify database creation:**
```powershell
ls -Path data\
```

You should see `expenses.db` file.

---

## Running the Application

### Step 1: Ensure Virtual Environment is Active

```powershell
.venv\Scripts\Activate.ps1
```

Check for `(.venv)` prefix in terminal.

### Step 2: Launch Streamlit App

Create `app.py` in project root (or use your existing app):

```python
import streamlit as st

st.set_page_config(
    page_title="Expense Tracker",
    page_icon="💰",
    layout="wide"
)

st.title("💰 AI-Enhanced Expense Tracker")
st.write("Track, categorize, and forecast your expenses with machine learning.")

# Placeholder content
st.info("Application initialized. Add your transaction data to get started.")
```

Then run:
```powershell
streamlit run app.py
```

**Expected output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501

  For better performance, install Watchdog:
  $ pip install watchdog
```

### Step 3: Access the Application

Open your browser and navigate to:
```
http://localhost:8501
```

You should see the Streamlit interface.

### Step 4: Stop the Application

In the terminal, press:
```
Ctrl + C
```

---

## Training Models

### Step 1: Prepare Training Data

Create `sample_transactions.csv` in the `data/` directory with labeled transactions:

```csv
date,amount,vendor,description,category
2025-05-03,42.75,Walmart,Grocery pickup - produce and dairy,Groceries
2025-05-05,3.25,Transit,Subway fare - downtown,Transportation
...
```

### Step 2: Create Training Script

Create `scripts/train_models.py`:

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pickle

def train_categorizer(df, model_path='models/categorizer.pkl'):
    """Train RandomForest categorization model."""
    # Feature engineering
    df['amount_log'] = np.log1p(df['amount'])
    df['desc_length'] = df['description'].str.len()
    df['word_count'] = df['description'].str.split().str.len()
    
    features = ['amount_log', 'desc_length', 'word_count']
    X = df[features]
    y = df['category']
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✓ Categorization model trained and saved")
    return model

if __name__ == '__main__':
    df = pd.read_csv('data/sample_transactions.csv')
    train_categorizer(df)
```

### Step 3: Run Training

```powershell
python scripts/train_models.py
```

Models will be saved in `models/` directory.

---

## Deployment to Streamlit Cloud

### Step 1: Push to GitHub

```powershell
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/expense-tracker.git
git push -u origin main
```

### Step 2: Create Streamlit Cloud Account

1. Visit https://streamlit.io/cloud
2. Click "Sign up with GitHub"
3. Authorize Streamlit to access your GitHub repositories

### Step 3: Deploy Application

1. In Streamlit Cloud dashboard, click "New app"
2. Select your repository: `yourusername/expense-tracker`
3. Select branch: `main`
4. Set main file path: `app.py`
5. Click "Deploy"

Streamlit will build and deploy your app. This takes 2-3 minutes.

### Step 4: Configure Secrets (Optional)

For sensitive data (API keys, database credentials):

1. In your deployed app, click "Manage" → "Settings"
2. Go to "Secrets" tab
3. Add your secrets in TOML format:
   ```toml
   [database]
   path = "sqlite:///expenses.db"
   
   [api]
   key = "your_api_key_here"
   ```

### Step 5: Access Deployed App

After deployment, your app will be live at:
```
https://yourusername-expense-tracker.streamlit.app
```

---

## Environment Variables

Create `.env` file in project root (for local development):

```
DATABASE_URL=sqlite:///data/expenses.db
DEBUG=False
LOG_LEVEL=INFO
```

Load in Python:
```python
from dotenv import load_dotenv
import os

load_dotenv()
db_url = os.getenv('DATABASE_URL')
```

---

## Troubleshooting

### Issue: Virtual environment won't activate

**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\Activate.ps1
```

### Issue: "python: command not found"

**Solution:**
- Ensure Python is in PATH
- Reinstall Python with "Add Python to PATH" checkbox enabled
- Use full path: `C:\Python311\python.exe`

### Issue: XGBoost installation fails on Windows

**Solution - Option 1:**
```powershell
pip install xgboost --only-binary :all:
```

**Solution - Option 2 (use Conda):**
```powershell
conda install xgboost
```

### Issue: "No module named 'streamlit'"

**Solution:**
```powershell
pip install -r requirements.txt
```

### Issue: Streamlit app won't start

**Solution:**
1. Verify virtual environment is active: `(.venv)` prefix in terminal
2. Check Python version: `python --version` (should be 3.9+)
3. Reinstall Streamlit: `pip install --upgrade streamlit`
4. Check for syntax errors: `python -m py_compile app.py`

### Issue: Database file not found

**Solution:**
```powershell
python scripts/init_db.py
```

This creates the database and schema.

### Issue: Models directory missing

**Solution:**
```powershell
mkdir models
mkdir data
mkdir scripts
```

### Issue: Permission denied when running script

**Solution:**
```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_models.py
```

### Issue: Streamlit Cloud deployment fails

**Solutions:**
1. Check `requirements.txt` is in repo root
2. Ensure `app.py` exists in repo root
3. Check for syntax errors: `python -m py_compile app.py`
4. View deployment logs in Streamlit Cloud dashboard
5. Ensure all imports in `app.py` are in `requirements.txt`

---

## Performance Tuning

### For Local Development
- Close other applications to free RAM
- Use smaller sample datasets initially (< 1000 rows)
- Enable Streamlit caching for expensive operations:
  ```python
  @st.cache_data
  def load_data():
      return pd.read_csv('data.csv')
  ```

### For Production Deployment
- Use Streamlit Cloud's multi-threaded deployment
- Set up database indexing for large tables
- Implement data pagination for dashboards
- Use model versioning and caching

---

## Next Steps

1. ✅ Development environment is ready
2. Explore `app.py` and customize your features
3. Add transaction data and train models
4. Test locally before deploying
5. Monitor performance on Streamlit Cloud

---

**Last Updated:** November 30, 2025  
**Version:** 1.0.0
