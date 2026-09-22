# AgroCare Week 3 Final

Week 3 UI + persistent SQLite backend + automatic scan history.

## Important
Copy your existing `models` folder (containing `best_model.joblib`) from the working Week 2 MultiCrop project into this folder.

## Run
1. `py -m pip install -r requirements.txt`
2. `py -m streamlit run app.py`

The SQLite database `agrocare.db` is created automatically after the app starts.

## Backend
Every completed supported or unknown diagnosis is automatically saved with:
Scan Date/Time, Image Name, Crop, Disease, Confidence, Severity, Status and Sharpness.

Scan History provides summary cards, persistent records, CSV export and Clear History.
