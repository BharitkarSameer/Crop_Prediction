# 🌾 Crop Recommendation System

An AI-powered web app that recommends the best crop to grow based on soil nutrients and climate conditions.

## Features
- Predicts the optimal crop from 22 types
- Shows confidence scores and top alternatives
- Clean, mobile-friendly UI

## Files
```
app.py              ← Main Streamlit application
RandomForest.pkl    ← Trained ML model (MultiOutputClassifier)
requirements.txt    ← Python dependencies
```

## Local Development
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🚀 Deploy Free on Streamlit Community Cloud

1. **Push this folder to a GitHub repo** (public or private)
   ```bash
   git init
   git add .
   git commit -m "Initial crop recommendation app"
   git remote add origin https://github.com/YOUR_USERNAME/crop-app.git
   git push -u origin main
   ```

2. **Go to** [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub

3. Click **"New app"** → Select your repo → Set:
   - Branch: `main`
   - Main file: `app.py`

4. Click **Deploy** — your app will be live in ~2 minutes at a free `.streamlit.app` URL!

## Input Parameters
| Parameter | Description | Unit |
|-----------|-------------|------|
| N | Nitrogen content | kg/ha |
| P | Phosphorus content | kg/ha |
| K | Potassium content | kg/ha |
| Temperature | Average temperature | °C |
| Humidity | Relative humidity | % |
| pH | Soil acidity | 0–14 |
| Rainfall | Average annual rainfall | mm |
