import streamlit as st
import numpy as np
import pickle
import os
import math
import warnings
import requests
import pandas as pd

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Recommendation",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: linear-gradient(135deg, #0d2b1a 0%, #1a4a2e 50%, #0a1f12 100%); min-height: 100vh; }
.hero-title { font-family: 'Playfair Display', serif; font-size: 2.8rem; color: #e8f5e3; text-align: center; margin-bottom: 0.2rem; letter-spacing: -0.02em; }
.hero-sub { text-align: center; color: #8ab89a; font-size: 1rem; font-weight: 300; margin-bottom: 2rem; letter-spacing: 0.08em; text-transform: uppercase; }
.card { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12); border-radius: 16px; padding: 1.6rem 2rem; margin-bottom: 1.2rem; backdrop-filter: blur(8px); }
.card-title { font-family: 'Playfair Display', serif; color: #b8e0c4; font-size: 1.1rem; margin-bottom: 1rem; border-bottom: 1px solid rgba(184,224,196,0.2); padding-bottom: 0.6rem; }
.result-box { background: linear-gradient(135deg, #1e5c34, #2d7a4a); border: 2px solid #4caf7d; border-radius: 20px; padding: 2rem; text-align: center; margin-top: 1.5rem; box-shadow: 0 8px 32px rgba(76,175,125,0.3); animation: fadeUp 0.5s ease; }
@keyframes fadeUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
.result-emoji { font-size: 3.5rem; margin-bottom: 0.5rem; }
.result-label { color: #aee8c4; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 0.3rem; }
.result-crop { font-family: 'Playfair Display', serif; font-size: 2.4rem; color: #ffffff; font-weight: 700; }
.result-conf { color: #7dd4a8; font-size: 0.9rem; margin-top: 0.4rem; }
.source-tag { display: inline-block; font-size: 0.7rem; padding: 0.15rem 0.5rem; border-radius: 20px; margin-left: 4px; vertical-align: middle; font-weight: 500; }
.src-api { background: rgba(76,175,125,0.2); color: #7dd4a8; border: 1px solid rgba(76,175,125,0.4); }
.src-district { background: rgba(255,193,7,0.2); color: #ffc107; border: 1px solid rgba(255,193,7,0.4); }
.src-fallback { background: rgba(255,100,100,0.15); color: #ff8a8a; border: 1px solid rgba(255,100,100,0.3); }
.src-manual { background: rgba(255,255,255,0.08); color: #aaa; border: 1px solid rgba(255,255,255,0.2); }
.badge { display: inline-block; background: rgba(76,175,125,0.2); border: 1px solid rgba(76,175,125,0.5); color: #7dd4a8; border-radius: 20px; padding: 0.2rem 0.8rem; font-size: 0.75rem; margin: 0.2rem; }
.top-crops { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.8rem; justify-content: center; }
.info-row { background: rgba(255,255,255,0.04); border-radius: 8px; padding: 0.5rem 0.8rem; margin: 0.3rem 0; font-size: 0.82rem; color: #8ab89a; }
div[data-testid="stNumberInput"] input { background: rgba(255,255,255,0.08) !important; border: 1px solid rgba(255,255,255,0.2) !important; border-radius: 8px !important; color: #e8f5e3 !important; }
div[data-testid="stNumberInput"] label { color: #8ab89a !important; font-size: 0.82rem !important; text-transform: uppercase; letter-spacing: 0.06em; }
.stButton > button { background: linear-gradient(135deg, #2d8a50, #3dab68) !important; color: white !important; border: none !important; border-radius: 12px !important; padding: 0.75rem 2.5rem !important; font-size: 1rem !important; font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important; letter-spacing: 0.04em !important; width: 100% !important; transition: all 0.2s ease !important; box-shadow: 0 4px 15px rgba(61,171,104,0.35) !important; }
.stButton > button:hover { background: linear-gradient(135deg, #3dab68, #4dc87a) !important; box-shadow: 0 6px 20px rgba(61,171,104,0.5) !important; transform: translateY(-1px) !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
LABELS = ['rice','maize','chickpea','kidneybeans','pigeonpeas','mothbeans',
          'mungbean','blackgram','lentil','pomegranate','banana','mango',
          'grapes','watermelon','muskmelon','apple','orange','papaya',
          'coconut','cotton','jute','coffee']

CROP_EMOJI = {
    'rice':'🌾','maize':'🌽','chickpea':'🫘','kidneybeans':'🫘','pigeonpeas':'🌿',
    'mothbeans':'🌱','mungbean':'🌱','blackgram':'🌱','lentil':'🫘','pomegranate':'🍎',
    'banana':'🍌','mango':'🥭','grapes':'🍇','watermelon':'🍉','muskmelon':'🍈',
    'apple':'🍎','orange':'🍊','papaya':'🍈','coconut':'🥥','cotton':'🌸',
    'jute':'🌿','coffee':'☕'
}
CROP_INFO = {
    'rice':'Thrives in waterlogged, warm conditions with high humidity.',
    'maize':'Versatile cereal crop adaptable to a wide range of soils.',
    'chickpea':'Drought-tolerant legume great for nitrogen fixation.',
    'kidneybeans':'Protein-rich legume requiring well-drained fertile soil.',
    'pigeonpeas':'Drought-resistant legume, excellent for dry regions.',
    'mothbeans':'Highly drought-tolerant crop suited for arid zones.',
    'mungbean':'Fast-growing legume ideal for tropical climates.',
    'blackgram':'Rich in protein, prefers warm humid weather.',
    'lentil':'Cool-season legume with excellent nutritional profile.',
    'pomegranate':'Hardy fruit crop that tolerates poor soils and drought.',
    'banana':'Tropical fruit needing high humidity and rich soil.',
    'mango':'Tropical fruit tree suited to hot, dry-season climates.',
    'grapes':'Requires well-drained soil and a dry, sunny climate.',
    'watermelon':'Warm-season crop needing sandy loam and plenty of sun.',
    'muskmelon':'Warm-climate fruit preferring sandy, well-drained soil.',
    'apple':'Temperate fruit requiring cool winters and mild summers.',
    'orange':'Citrus fruit thriving in subtropical, frost-free climates.',
    'papaya':'Fast-growing tropical fruit needing warmth and moisture.',
    'coconut':'Coastal tropical crop loving sandy soil and high humidity.',
    'cotton':'Fiber crop needing long frost-free seasons and deep soils.',
    'jute':'Fiber crop thriving in warm, humid, waterlogged conditions.',
    'coffee':'Shade-loving tropical crop needing rich volcanic soil.',
}

# Approximate district centroids for nearest-match lookup
DISTRICT_COORDS = {
    "Anantapur":(14.68,77.60),"Chittoor":(13.21,79.10),"East Godavari":(17.32,82.00),
    "Guntur":(16.30,80.44),"Krishna":(16.61,80.52),"Kurnool":(15.83,78.04),
    "Nellore":(14.44,79.99),"Prakasam":(15.34,79.70),"Srikakulam":(18.30,83.90),
    "Visakhapatnam":(17.69,83.21),"Vizianagaram":(18.12,83.40),"West Godavari":(16.91,81.35),
    "YSR Kadapa":(14.46,78.82),"Barpeta":(26.32,91.00),"Cachar":(24.80,92.75),
    "Dibrugarh":(27.48,94.90),"Jorhat":(26.75,94.21),"Kamrup":(26.18,91.74),
    "Nagaon":(26.35,92.69),"Tinsukia":(27.49,95.36),"Araria":(26.14,87.52),
    "Bhagalpur":(25.24,87.01),"Darbhanga":(26.16,85.90),"Gaya":(24.80,85.00),
    "Muzaffarpur":(26.12,85.37),"Patna":(25.60,85.14),"Nalanda":(25.11,85.44),
    "Samastipur":(25.86,85.78),"Vaishali":(25.69,85.21),"Bhopal":(23.25,77.40),
    "Indore":(22.72,75.86),"Jabalpur":(23.17,79.95),"Gwalior":(26.22,78.18),
    "Raipur":(21.25,81.63),"Bilaspur":(22.09,82.13),"Mumbai City":(18.96,72.82),
    "Mumbai Suburban":(19.13,72.85),"Pune":(18.52,73.86),"Nagpur":(21.15,79.09),
    "Nashik":(20.00,73.78),"Aurangabad":(19.88,75.33),"Kolhapur":(16.70,74.23),
    "Amravati":(20.93,77.75),"Solapur":(17.68,75.90),"Thane":(19.21,72.97),
    "Imphal East":(24.82,93.97),"Imphal West":(24.80,93.90),
    "Aizawl":(23.73,92.72),"Kohima":(25.67,94.11),"Dimapur":(25.90,93.72),
    "Cuttack":(20.46,85.88),"Puri":(19.81,85.83),"Ganjam":(19.38,84.88),
    "Sambalpur":(21.47,83.97),"Sundargarh":(22.12,84.04),
    "Amritsar":(31.63,74.87),"Ludhiana":(30.90,75.85),"Patiala":(30.34,76.37),
    "Jalandhar":(31.33,75.57),"Gurdaspur":(32.04,75.40),"Hoshiarpur":(31.53,75.91),
    "Jaipur":(26.91,75.79),"Jodhpur":(26.29,73.02),"Udaipur":(24.59,73.69),
    "Ajmer":(26.45,74.64),"Bikaner":(28.02,73.31),"Kota":(25.18,75.84),
    "Alwar":(27.56,76.61),"Barmer":(25.75,71.39),"Jaisalmer":(26.92,70.90),
    "Chennai":(13.08,80.27),"Coimbatore":(11.02,76.97),"Madurai":(9.93,78.12),
    "Salem":(11.66,78.14),"Tiruchirappalli":(10.79,78.70),"Vellore":(12.91,79.13),
    "Thanjavur":(10.80,79.14),"Erode":(11.34,77.72),"Tiruppur":(11.10,77.34),
    "Hyderabad":(17.38,78.47),"Warangal Urban":(17.97,79.59),"Karimnagar":(18.43,79.13),
    "Nizamabad":(18.67,78.09),"Khammam":(17.24,80.15),"Nalgonda":(17.05,79.27),
    "Agra":(27.18,78.01),"Lucknow":(26.85,80.95),"Kanpur Nagar":(26.45,80.33),
    "Varanasi":(25.32,83.00),"Allahabad":(25.44,81.84),"Meerut":(28.98,77.71),
    "Bareilly":(28.36,79.41),"Moradabad":(28.83,78.78),"Saharanpur":(29.97,77.55),
    "Gorakhpur":(26.75,83.37),"Mathura":(27.49,77.67),"Ghaziabad":(28.67,77.45),
    "Dehradun":(30.32,78.03),"Haridwar":(29.95,78.16),"Nainital":(29.38,79.46),
    "Almora":(29.60,79.66),"Kolkata":(22.57,88.36),"Darjeeling":(27.04,88.27),
    "Jalpaiguri":(26.54,88.73),"Malda":(25.00,88.13),"Murshidabad":(24.18,88.25),
    "Birbhum":(23.88,87.53),"Bankura":(23.22,87.07),"Purba Medinipur":(22.22,87.73),
}

# ── Load model & district data ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), "RandomForest.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_district_data():
    path = os.path.join(os.path.dirname(__file__), "india_soil_data.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

model = load_model()
district_df = load_district_data()

# ── Helpers ────────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def nearest_district(lat, lon):
    if district_df is None:
        return None, None, float('inf')
    best_row, best_dist = None, float('inf')
    for _, row in district_df.iterrows():
        name = row['district']
        if name in DISTRICT_COORDS:
            dlat, dlon = DISTRICT_COORDS[name]
            d = haversine(lat, lon, dlat, dlon)
            if d < best_dist:
                best_dist = d
                best_row = row
    return best_row, f"{best_row['district']}, {best_row['state']}" if best_row is not None else None, best_dist

def fetch_climate(lat, lon):
    """Layer 1: Open-Meteo live climate data."""
    try:
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
               f"&current=temperature_2m,relative_humidity_2m"
               f"&daily=precipitation_sum&timezone=auto&forecast_days=7")
        r = requests.get(url, timeout=6)
        d = r.json()
        temp = round(d['current']['temperature_2m'], 1)
        hum  = round(d['current']['relative_humidity_2m'], 1)
        rain = round(sum(d['daily']['precipitation_sum']) / 7 * 30, 1)
        return temp, hum, rain, "api"
    except Exception:
        return None, None, None, "failed"

def fetch_ph_openlandmap(lat, lon):
    """Layer 1: OpenLandMap 250m resolution pH."""
    try:
        url = (f"http://api.openlandmap.org/query/point"
               f"?lat={lat}&lon={lon}&coll=predicted250m"
               f"&regex=sol_ph.h2o_usda.4c1a2a_m_250m_b0..0cm_1950..2017_v0.2.tif")
        r = requests.get(url, timeout=8)
        data = r.json()
        resp = data.get('response', [])
        if resp:
            ph_key = next((k for k in resp[0] if 'ph' in k.lower()), None)
            if ph_key:
                ph = round(resp[0][ph_key] / 10, 1)   # stored as pH × 10
                if 3.0 <= ph <= 10.0:
                    return ph, "api"
        return None, "failed"
    except Exception:
        return None, "failed"

def get_auto_values(lat, lon):
    """
    3-layer auto-fill system:
      Temp / Humidity / Rainfall → Layer 1: Open-Meteo  |  Layer 3: national avg
      pH                         → Layer 1: OpenLandMap  |  Layer 2: district CSV  |  Layer 3: national avg
      N / P / K                  → Layer 2: district CSV |  Layer 3: national avg
    """
    results, sources = {}, {}

    # Climate
    temp, hum, rain, clim_src = fetch_climate(lat, lon)
    if clim_src == "api":
        results.update({'temperature': temp, 'humidity': hum, 'rainfall': rain})
        sources.update({'temperature':'api', 'humidity':'api', 'rainfall':'api'})
    else:
        results.update({'temperature': 25.0, 'humidity': 65.0, 'rainfall': 100.0})
        sources.update({'temperature':'fallback', 'humidity':'fallback', 'rainfall':'fallback'})

    # pH
    ph, ph_src = fetch_ph_openlandmap(lat, lon)
    if ph_src == "api":
        results['ph'] = ph
        sources['ph'] = 'api'

    # N, P, K + pH fallback: nearest district
    row, district_label, dist_km = nearest_district(lat, lon)
    if row is not None:
        results.update({'N': int(row['N']), 'P': int(row['P']), 'K': int(row['K'])})
        sources.update({'N': 'district', 'P': 'district', 'K': 'district'})
        if ph_src != "api":
            results['ph'] = round(float(row['ph']), 1)
            sources['ph'] = 'district'
    else:
        # national averages
        results.update({'N': 50, 'P': 53, 'K': 48})
        sources.update({'N': 'fallback', 'P': 'fallback', 'K': 'fallback'})
        if ph_src != "api":
            results['ph'] = 6.5
            sources['ph'] = 'fallback'
        district_label = "India (national average)"
        dist_km = None

    return results, sources, district_label, dist_km

def source_badge(src):
    labels = {
        'api':      ('src-api',      '📡 Live API'),
        'district': ('src-district', '📍 District avg'),
        'fallback': ('src-fallback', '📊 National avg'),
        'manual':   ('src-manual',   '✏️ Manual'),
    }
    cls, txt = labels.get(src, ('src-manual', '✏️ Manual'))
    return f'<span class="source-tag {cls}">{txt}</span>'

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    sample = np.array([[N, P, K, temperature, humidity, ph, rainfall]], dtype=float)
    probs  = np.array([arr[0][1] for arr in model.predict_proba(sample)])
    top5   = [(LABELS[i], float(probs[i])) for i in np.argsort(probs)[::-1][:5]]
    return top5[0][0], top5[0][1], top5

# ── Session state defaults ─────────────────────────────────────────────────────
field_defaults = {'N':90.0,'P':42.0,'K':43.0,'temperature':25.0,'humidity':80.0,'ph':6.5,'rainfall':200.0}
for k, v in field_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
if 'sources' not in st.session_state:
    st.session_state['sources'] = {k:'manual' for k in field_defaults}
if 'location_info' not in st.session_state:
    st.session_state['location_info'] = None

# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🌾 Crop Recommendation</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI-powered soil & climate analysis</div>', unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model file `RandomForest.pkl` not found.")
    st.stop()

# ────────────────────────────────────────────────────────────────
# AUTO-DETECT LOCATION VIA QUERY PARAMS (JS → URL → Streamlit)
# ────────────────────────────────────────────────────────────────
params = st.query_params
if "lat" in params and "lon" in params and "gps_done" not in st.session_state:
    try:
        detected_lat = float(params["lat"])
        detected_lon = float(params["lon"])
        st.session_state["detected_lat"] = detected_lat
        st.session_state["detected_lon"] = detected_lon
        st.session_state["gps_done"] = True
        # Auto-trigger fill immediately
        with st.spinner("📡 Location detected! Fetching soil & climate data..."):
            vals, srcs, district_label, dist_km = get_auto_values(detected_lat, detected_lon)
            for k in field_defaults:
                st.session_state[k] = float(vals[k])
                st.session_state['sources'][k] = srcs[k]
            dist_str = f" (~{dist_km:.0f} km away)" if dist_km else ""
            st.session_state['location_info'] = (
                f"📍 Nearest reference district: **{district_label}**{dist_str}\n\n"
                f"Sources — Temperature/Humidity/Rainfall: {srcs.get('temperature','?').upper()} &nbsp;|&nbsp; "
                f"pH: {srcs.get('ph','?').upper()} &nbsp;|&nbsp; N/P/K: {srcs.get('N','?').upper()}"
            )
        st.query_params.clear()   # clean the URL after reading
        st.rerun()
    except Exception:
        pass

# ────────────────────────────────────────────────────────────────
# LOCATION CARD
# ────────────────────────────────────────────────────────────────
detected_lat = st.session_state.get("detected_lat", 20.5937)
detected_lon = st.session_state.get("detected_lon", 78.9629)
# Always sync session state keys so number_input reflects detected coords
st.session_state["lat_field"] = float(detected_lat)
st.session_state["lon_field"] = float(detected_lon)

st.markdown('<div class="card"><div class="card-title">📍 Auto-fill From Your Location</div>', unsafe_allow_html=True)
st.markdown('<p style="color:#8ab89a;font-size:0.88rem;margin-bottom:0.8rem;">Click the button below to detect your location automatically, or enter coordinates manually.</p>', unsafe_allow_html=True)

# JS geolocation → writes lat/lon into URL query params → Streamlit re-runs and reads them
st.components.v1.html("""
<style>
  #loc-btn {
    background: linear-gradient(135deg, #1a5c38, #2d8a50);
    color: white; border: 1px solid #4caf7d;
    border-radius: 12px; padding: 0.65rem 2rem;
    font-size: 0.95rem; cursor: pointer;
    font-family: sans-serif; width: 100%;
    box-shadow: 0 4px 15px rgba(76,175,125,0.3);
    transition: all 0.2s ease;
  }
  #loc-btn:hover { background: linear-gradient(135deg, #2d8a50, #3dab68); }
  #loc-btn:disabled { opacity: 0.6; cursor: not-allowed; }
  #loc-status { color: #8ab89a; font-size: 0.82rem; margin-top: 0.5rem; text-align: center; font-family: sans-serif; min-height: 1.2rem; }
</style>
<button id="loc-btn" onclick="getLocation()">📍 Detect My Location Automatically</button>
<div id="loc-status"></div>
<script>
function getLocation() {
  var btn = document.getElementById('loc-btn');
  var status = document.getElementById('loc-status');
  if (!navigator.geolocation) {
    status.textContent = '⚠️ Geolocation not supported. Please enter coordinates manually.';
    return;
  }
  btn.disabled = true;
  btn.textContent = '🔄 Detecting location...';
  status.textContent = 'Please allow location access when prompted by your browser.';
  navigator.geolocation.getCurrentPosition(
    function(pos) {
      var lat = pos.coords.latitude.toFixed(5);
      var lon = pos.coords.longitude.toFixed(5);
      status.textContent = '✅ Location found (' + lat + ', ' + lon + ') — loading data...';
      btn.textContent = '✅ Location Detected!';
      // Write coords into URL query params — Streamlit will re-run and pick them up
      var url = window.parent.location.href.split('?')[0];
      window.parent.location.href = url + '?lat=' + lat + '&lon=' + lon;
    },
    function(err) {
      btn.disabled = false;
      btn.textContent = '📍 Detect My Location Automatically';
      if (err.code === 1) {
        status.textContent = '❌ Location access denied. Please enter coordinates manually below.';
      } else if (err.code === 2) {
        status.textContent = '❌ Location unavailable. Please enter coordinates manually.';
      } else {
        status.textContent = '❌ Timed out. Please enter coordinates manually.';
      }
    },
    { timeout: 12000, enableHighAccuracy: true }
  );
}
</script>
""", height=90)

st.markdown('<p style="color:#6a9e7a;font-size:0.78rem;margin-top:0.5rem;margin-bottom:0.8rem;">💡 Or enter coordinates manually — open Google Maps, long-press your farm, and the lat/lon appears at the top.</p>', unsafe_allow_html=True)

col_lat, col_lon = st.columns(2)
with col_lat:
    lat_input = st.number_input("Latitude", value=st.session_state["lat_field"], format="%.4f", step=0.0001, key="lat_field")
with col_lon:
    lon_input = st.number_input("Longitude", value=st.session_state["lon_field"], format="%.4f", step=0.0001, key="lon_field")

if st.button("🌍 Auto-fill Soil & Climate Data"):
    with st.spinner("Fetching live data for your location..."):
        vals, srcs, district_label, dist_km = get_auto_values(lat_input, lon_input)
        for k in field_defaults:
            st.session_state[k] = float(vals[k])
            st.session_state['sources'][k] = srcs[k]
        dist_str = f" (~{dist_km:.0f} km away)" if dist_km else ""
        st.session_state['location_info'] = (
            f"📍 Nearest reference district: **{district_label}**{dist_str}\n\n"
            f"Sources — Temperature/Humidity/Rainfall: {srcs.get('temperature','?').upper()} &nbsp;|&nbsp; "
            f"pH: {srcs.get('ph','?').upper()} &nbsp;|&nbsp; N/P/K: {srcs.get('N','?').upper()}"
        )
        st.rerun()

if st.session_state['location_info']:
    st.markdown(f'<div class="info-row">{st.session_state["location_info"]}</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.73rem;color:#6a9e7a;margin-top:0.4rem;line-height:1.8rem;">
      <span class="source-tag src-api">📡 Live API</span> fetched live from satellites/sensors &nbsp;·&nbsp;
      <span class="source-tag src-district">📍 District avg</span> regional average from our India soil database &nbsp;·&nbsp;
      <span class="source-tag src-fallback">📊 National avg</span> India-wide average (API unavailable) &nbsp;·&nbsp;
      <span class="source-tag src-manual">✏️ Manual</span> entered by you
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────
# SOIL NUTRIENTS CARD
# ────────────────────────────────────────────────────────────────
srcs = st.session_state['sources']
n_badge = source_badge(srcs.get('N','manual'))
st.markdown(f'<div class="card"><div class="card-title">🧪 Soil Nutrients &nbsp;{n_badge}</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    N = st.number_input("Nitrogen (N) kg/ha", 0.0, 300.0, float(st.session_state['N']), 1.0)
with col2:
    P = st.number_input("Phosphorus (P) kg/ha", 0.0, 300.0, float(st.session_state['P']), 1.0)
with col3:
    K = st.number_input("Potassium (K) kg/ha", 0.0, 300.0, float(st.session_state['K']), 1.0)
st.markdown('</div>', unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────
# CLIMATE CARD
# ────────────────────────────────────────────────────────────────
clim_badge = source_badge(srcs.get('temperature','manual'))
st.markdown(f'<div class="card"><div class="card-title">🌤️ Climate & Soil Conditions &nbsp;{clim_badge}</div>', unsafe_allow_html=True)
col4, col5 = st.columns(2)
with col4:
    temperature = st.number_input("Temperature (°C)", -10.0, 55.0, float(st.session_state['temperature']), 0.5)
    humidity    = st.number_input("Humidity (%)",       0.0, 100.0, float(st.session_state['humidity']),    1.0)
with col5:
    ph_badge = source_badge(srcs.get('ph','manual'))
    st.markdown(f'<div style="margin-bottom:4px;font-size:0.8rem;color:#8ab89a;text-transform:uppercase;letter-spacing:0.06em;">Soil pH &nbsp;{ph_badge}</div>', unsafe_allow_html=True)
    ph       = st.number_input("pH", 0.0, 14.0, float(st.session_state['ph']), 0.1, label_visibility="collapsed")
    rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, float(st.session_state['rainfall']), 5.0)
st.markdown('</div>', unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────
# PREDICT
# ────────────────────────────────────────────────────────────────
if st.button("✨ Recommend Best Crop"):
    top_crop, confidence, top5 = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
    emoji    = CROP_EMOJI.get(top_crop, "🌱")
    info     = CROP_INFO.get(top_crop, "")
    conf_pct = min(confidence * 100, 100)
    conf_str = f"{conf_pct:.0f}% confidence" if conf_pct >= 1 else "Low confidence — try adjusting inputs"

    st.markdown(f"""
    <div class="result-box">
        <div class="result-emoji">{emoji}</div>
        <div class="result-label">Recommended Crop</div>
        <div class="result-crop">{top_crop.title()}</div>
        <div class="result-conf">{conf_str}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div style='color:#8ab89a;font-size:0.9rem;text-align:center;margin-top:0.8rem;'>{info}</div>",
                unsafe_allow_html=True)

    others = [(c, p) for c, p in top5[1:] if p > 0.01]
    if others:
        st.markdown("<div style='text-align:center;color:#6a9e7a;font-size:0.8rem;margin-top:1.2rem;text-transform:uppercase;letter-spacing:0.1em;'>Other Suitable Crops</div>", unsafe_allow_html=True)
        badges = "".join([f'<span class="badge">{CROP_EMOJI.get(c,"🌱")} {c.title()} ({p*100:.0f}%)</span>' for c, p in others])
        st.markdown(f'<div class="top-crops">{badges}</div>', unsafe_allow_html=True)
