import streamlit as st
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Recommendation",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0d2b1a 0%, #1a4a2e 50%, #0a1f12 100%);
    min-height: 100vh;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    color: #e8f5e3;
    text-align: center;
    margin-bottom: 0.2rem;
    letter-spacing: -0.02em;
}

.hero-sub {
    text-align: center;
    color: #8ab89a;
    font-size: 1rem;
    font-weight: 300;
    margin-bottom: 2.5rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(8px);
}

.card-title {
    font-family: 'Playfair Display', serif;
    color: #b8e0c4;
    font-size: 1.1rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid rgba(184,224,196,0.2);
    padding-bottom: 0.6rem;
}

.result-box {
    background: linear-gradient(135deg, #1e5c34, #2d7a4a);
    border: 2px solid #4caf7d;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
    box-shadow: 0 8px 32px rgba(76,175,125,0.3);
    animation: fadeUp 0.5s ease;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

.result-emoji { font-size: 3.5rem; margin-bottom: 0.5rem; }

.result-label {
    color: #aee8c4;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.3rem;
}

.result-crop {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    color: #ffffff;
    font-weight: 700;
}

.result-conf {
    color: #7dd4a8;
    font-size: 0.9rem;
    margin-top: 0.4rem;
}

.param-label {
    color: #8ab89a;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.2rem;
}

/* Override Streamlit input styling */
div[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 8px !important;
    color: #e8f5e3 !important;
}

div[data-testid="stNumberInput"] label {
    color: #8ab89a !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.stButton > button {
    background: linear-gradient(135deg, #2d8a50, #3dab68) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2.5rem !important;
    font-size: 1rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(61,171,104,0.35) !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #3dab68, #4dc87a) !important;
    box-shadow: 0 6px 20px rgba(61,171,104,0.5) !important;
    transform: translateY(-1px) !important;
}

.badge {
    display: inline-block;
    background: rgba(76,175,125,0.2);
    border: 1px solid rgba(76,175,125,0.5);
    color: #7dd4a8;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.75rem;
    margin: 0.2rem;
}

.top-crops {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.8rem;
    justify-content: center;
}

div[data-testid="stAlert"] {
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Crop metadata ─────────────────────────────────────────────────────────────
LABELS = [
    'rice','maize','chickpea','kidneybeans','pigeonpeas',
    'mothbeans','mungbean','blackgram','lentil','pomegranate',
    'banana','mango','grapes','watermelon','muskmelon','apple',
    'orange','papaya','coconut','cotton','jute','coffee'
]

CROP_EMOJI = {
    'rice':'🌾','maize':'🌽','chickpea':'🫘','kidneybeans':'🫘','pigeonpeas':'🌿',
    'mothbeans':'🌱','mungbean':'🌱','blackgram':'🌱','lentil':'🫘','pomegranate':'🍎',
    'banana':'🍌','mango':'🥭','grapes':'🍇','watermelon':'🍉','muskmelon':'🍈',
    'apple':'🍎','orange':'🍊','papaya':'🍈','coconut':'🥥','cotton':'🌸',
    'jute':'🌿','coffee':'☕'
}

CROP_INFO = {
    'rice': 'Thrives in waterlogged, warm conditions with high humidity.',
    'maize': 'Versatile cereal crop adaptable to a wide range of soils.',
    'chickpea': 'Drought-tolerant legume great for nitrogen fixation.',
    'kidneybeans': 'Protein-rich legume requiring well-drained fertile soil.',
    'pigeonpeas': 'Drought-resistant legume, excellent for dry regions.',
    'mothbeans': 'Highly drought-tolerant crop suited for arid zones.',
    'mungbean': 'Fast-growing legume ideal for tropical climates.',
    'blackgram': 'Rich in protein, prefers warm humid weather.',
    'lentil': 'Cool-season legume with excellent nutritional profile.',
    'pomegranate': 'Hardy fruit crop that tolerates poor soils and drought.',
    'banana': 'Tropical fruit needing high humidity and rich soil.',
    'mango': 'Tropical fruit tree suited to hot, dry-season climates.',
    'grapes': 'Requires well-drained soil and a dry, sunny climate.',
    'watermelon': 'Warm-season crop needing sandy loam and plenty of sun.',
    'muskmelon': 'Warm-climate fruit preferring sandy, well-drained soil.',
    'apple': 'Temperate fruit requiring cool winters and mild summers.',
    'orange': 'Citrus fruit thriving in subtropical, frost-free climates.',
    'papaya': 'Fast-growing tropical fruit needing warmth and moisture.',
    'coconut': 'Coastal tropical crop loving sandy soil and high humidity.',
    'cotton': 'Fiber crop needing long frost-free seasons and deep soils.',
    'jute': 'Fiber crop thriving in warm, humid, waterlogged conditions.',
    'coffee': 'Shade-loving tropical crop needing rich volcanic soil.',
}

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "RandomForest.pkl")
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()

# ── Prediction function ────────────────────────────────────────────────────────
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    """Returns (top_crop, confidence, top5_list_of_(crop, prob))."""
    sample = np.array([[N, P, K, temperature, humidity, ph, rainfall]], dtype=float)
    proba_list = model.predict_proba(sample)   # list of 22 arrays, each (1,2)
    probs = np.array([arr[0][1] for arr in proba_list])  # probability of True per label
    
    top_idx = np.argsort(probs)[::-1][:5]
    top5 = [(LABELS[i], float(probs[i])) for i in top_idx]
    
    best_label, best_prob = top5[0]
    return best_label, best_prob, top5

# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🌾 Crop Recommendation</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI-powered soil & climate analysis</div>', unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model file `RandomForest.pkl` not found. Make sure it's in the same folder as `app.py`.")
    st.stop()

# Soil Nutrients
st.markdown('<div class="card"><div class="card-title">🧪 Soil Nutrients</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    N = st.number_input("Nitrogen (N) kg/ha", min_value=0.0, max_value=200.0, value=90.0, step=1.0)
with col2:
    P = st.number_input("Phosphorus (P) kg/ha", min_value=0.0, max_value=200.0, value=42.0, step=1.0)
with col3:
    K = st.number_input("Potassium (K) kg/ha", min_value=0.0, max_value=200.0, value=43.0, step=1.0)
st.markdown('</div>', unsafe_allow_html=True)

# Climate
st.markdown('<div class="card"><div class="card-title">🌤️ Climate Conditions</div>', unsafe_allow_html=True)
col4, col5 = st.columns(2)
with col4:
    temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=55.0, value=25.0, step=0.5)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
with col5:
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=200.0, step=5.0)
st.markdown('</div>', unsafe_allow_html=True)

# Predict button
predict_clicked = st.button("✨ Recommend Best Crop")

if predict_clicked:
    top_crop, confidence, top5 = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
    emoji = CROP_EMOJI.get(top_crop, "🌱")
    info = CROP_INFO.get(top_crop, "")

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

    st.markdown(f"<div style='color:#8ab89a; font-size:0.9rem; text-align:center; margin-top:0.8rem;'>{info}</div>",
                unsafe_allow_html=True)

    # Top 5 alternatives
    others = [(c, p) for c, p in top5[1:] if p > 0.01]
    if others:
        st.markdown("<div style='text-align:center; color:#6a9e7a; font-size:0.8rem; margin-top:1.2rem; text-transform:uppercase; letter-spacing:0.1em;'>Other Suitable Crops</div>", unsafe_allow_html=True)
        badges = "".join([
            f'<span class="badge">{CROP_EMOJI.get(c,"🌱")} {c.title()} ({p*100:.0f}%)</span>'
            for c, p in others
        ])
        st.markdown(f'<div class="top-crops">{badges}</div>', unsafe_allow_html=True)
