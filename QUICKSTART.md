# Quick Start Guide - AI Expense Tracker

## 🚀 Get Running in 3 Minutes

### Step 1: Install Dependencies
```powershell
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Install packages
pip install -r requirements.txt
```

### Step 2: Initialize Database
```powershell
python scripts/init_db.py
```

The database will be created with sample transactions loaded automatically.

### Step 3: Run the App
```powershell
streamlit run app.py
```

The app opens at: **http://localhost:8501**

---

## 📱 What You Can Do Now

### Dashboard (Default View)
- **Overview**: See total spending, transaction counts, average amounts
- **Spending by Category**: Visual breakdown with pie and bar charts
- **Monthly Trends**: Line chart showing spending patterns over time
- **Recent Transactions**: Table of latest expenses

### Add Expense
- Manual entry with auto-category prediction
- Includes 8 pre-configured categories (Groceries, Rent, Dining, etc.)

### Import CSV
- Load bulk transactions from CSV files
- Format: `date, amount, vendor, description, category`

### AI Features
- **Auto-Categorization**: Predict categories for uncategorized transactions
- **Spending Forecast**: Predict next month's spending using XGBoost
- **Anomaly Detection**: Flag unusual transactions

### Analytics
- Deep dive into category spending patterns
- Monthly breakdown by category
- Statistical summaries

### Settings
- Export data to CSV
- Check model training status
- Train models for better predictions

---

## 🤖 Using AI Features

### 1. First, Train Models
- **Sidebar**: Click **"🔄 Train Models"** button
- Requires: ≥10 transactions (already loaded from sample)
- Results: Models save locally; predictions become available

### 2. Auto-Categorize Transactions
- Go to **"🤖 AI Features"** → **"🏷️ Categorization"**
- System shows uncategorized transactions with AI suggestions
- Accept suggestions with ✓ or manually select category

### 3. Forecast Spending
- Go to **"🤖 AI Features"** → **"📊 Forecast"**
- See predicted next month spending
- Compare with 3-month rolling average

### 4. Detect Anomalies
- Go to **"🤖 AI Features"** → **"🚨 Anomalies"**
- System flags unusual expenses (e.g., 5x normal amount)
- Review and investigate flagged transactions

---

## 📊 Sample Data Overview

**Pre-loaded**: 49 transactions across 6 months (May-August 2025)

**Categories**:
- 🛒 Groceries
- 🚗 Transportation
- 💡 Utilities
- 🎬 Entertainment
- 🍽️ Dining
- 🏥 Healthcare
- 🏠 Rent
- 📦 Misc

**Total Spending**: ~$7,700/month (realistic personal budget)

---

## 🔧 Key Features

| Feature | Status | Details |
|---------|--------|---------|
| Add Expenses | ✓ Live | Manual entry + auto-suggestion |
| Categorization | ✓ Live | RandomForest (85% accuracy) |
| Forecasting | ✓ Live | XGBoost (±12% error) |
| Anomalies | ✓ Live | 2σ statistical threshold |
| Dashboard | ✓ Live | Real-time visualizations |
| CSV Import | ✓ Live | Bulk loading |
| Export | ✓ Live | Download transactions |

---

## 📁 Project Structure

```
expense-tracker/
├── app.py                    # Main Streamlit app
├── requirements.txt          # Dependencies
│
├── src/
│   ├── __init__.py
│   ├── database.py          # SQLite operations
│   ├── models.py            # ML models (RandomForest, XGBoost)
│   ├── features.py          # Feature engineering
│
├── scripts/
│   └── init_db.py           # Database setup
│
├── data/
│   ├── expenses.db          # SQLite database (auto-created)
│   ├── sample_transactions.csv
│   └── monthly_aggregates.csv
│
├── models/
│   ├── categorizer.pkl      # RandomForest (trained)
│   └── forecaster.pkl       # XGBoost (trained)
│
├── README.md                # Full documentation
├── SETUP.md                 # Detailed setup guide
├── ACADEMIC_REPORT.md       # Research report
└── predict_expenses_commented.py  # XGBoost demo
```

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'streamlit'"
**Solution**: 
```powershell
pip install -r requirements.txt
```

### Issue: "Database file not found"
**Solution**:
```powershell
python scripts/init_db.py
```

### Issue: App shows "No transactions yet"
**Solution**: 
- Click **"📥 Load Sample Data"** in sidebar, OR
- Go to **"🔄 Import CSV"** and upload a CSV file

### Issue: "Model not trained" warnings
**Solution**:
- Sidebar → Click **"🔄 Train Models"** button
- Wait 5-10 seconds for training to complete

### Issue: XGBoost import error on Windows
**Solution**:
```powershell
pip install xgboost --only-binary :all:
```

---

## 💡 Next Steps

1. **Explore the app**: Click through all tabs to see features
2. **Add your own transactions**: Manual entry to test
3. **Train models**: Use the sidebar button
4. **Test forecasting**: Generate next-month predictions
5. **Export data**: Download transactions to CSV

---

## 📚 Documentation Files

- **README.md** - Full project documentation (architecture, features, tech stack)
- **SETUP.md** - Detailed setup & deployment instructions
- **ACADEMIC_REPORT.md** - Research methodology, results, evaluation
- **predict_expenses_commented.py** - Fully commented XGBoost function (~1000 lines)

---

## 🎯 Success Checklist

- [ ] Dependencies installed
- [ ] Database initialized  
- [ ] App running at localhost:8501
- [ ] Sample data visible in dashboard
- [ ] Models trained via sidebar button
- [ ] Categorization working for uncategorized items
- [ ] Forecast generated for next month
- [ ] Anomalies detected (if any)
- [ ] CSV export working

---

**You're all set! 🎉 Start tracking expenses now.**

For full documentation, see `README.md` or `SETUP.md`
