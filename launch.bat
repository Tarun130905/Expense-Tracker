@echo off
REM Launch script for AI-Enhanced Expense Tracker
REM Activates virtual environment and starts Streamlit app

echo.
echo =========================================================
echo   AI-Enhanced Expense Tracker
echo   Launching application...
echo =========================================================
echo.

REM Activate virtual environment
call .venv\Scripts\activate.bat

if %errorlevel% neq 0 (
    echo Error: Virtual environment not found!
    echo Please run: python -m venv .venv
    echo Then run: pip install -r requirements.txt
    pause
    exit /b 1
)

echo [OK] Virtual environment activated
echo.

REM Check if database exists
if not exist "data\expenses.db" (
    echo [INFO] Database not found. Initializing...
    python scripts/init_db.py
    if %errorlevel% neq 0 (
        echo Error initializing database!
        pause
        exit /b 1
    )
    echo [OK] Database initialized
    echo.
)

REM Start Streamlit app
echo [INFO] Starting Streamlit app...
echo [INFO] Opening browser to http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

streamlit run app.py

pause
