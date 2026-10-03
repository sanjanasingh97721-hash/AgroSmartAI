import streamlit as st
import pickle
import numpy as np

# ── Page Configuration ────────────────────────────────────────
st.set_page_config(
    page_title="AgroSmart AI — Crop Recommendation",
    page_icon="🌾",
    layout="wide"
)

# ── Load Model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model/crop_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ── CSS Styling ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp { background: #f5f8f3; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

[data-testid="stSidebar"] { background: #174d2a; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label { color: white !important; }

.project-title   { font-size: 38px; font-weight: 800; color: #174d2a; margin-bottom: 5px; }
.project-subtitle{ font-size: 17px; color: #687568; margin-bottom: 25px; }
.section-heading { font-size: 22px; font-weight: 700; color: #174d2a; margin: 22px 0 10px 0; }

[data-testid="stNumberInput"] {
    background: white; padding: 12px;
    border-radius: 14px; border: 1px solid #e0e8df;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}
[data-testid="stNumberInput"] label { color: #315b3b !important; font-weight: 650 !important; }

.stButton > button {
    width: 100%; height: 55px; border-radius: 12px;
    border: none; background: #218c45;
    color: white !important; font-size: 17px; font-weight: 700;
    transition: background 0.2s;
}
.stButton > button:hover { background: #176f35; }

.result-box {
    background: #e8f5e9; border: 2px solid #75b879;
    border-radius: 18px; padding: 28px;
    text-align: center; margin-top: 25px;
}
.result-label { color: #52705a; font-size: 16px; font-weight: 600; }
.result-crop  { color: #176f35; font-size: 42px; font-weight: 800; margin-top: 8px; }
.footer-text  { text-align: center; color: #748074; margin-top: 40px; padding: 20px; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🌾")
    st.title("AgroSmart AI")
    st.write("Machine Learning based crop recommendation system for precision agriculture.")
    st.markdown("---")
    st.subheader("Model Info")
    st.write("🌲 **Algorithm:** Random Forest")
    st.write("📊 **Task:** Multi-class Classification")
    st.write("🔢 **Features:** 7 (NPK + Climate)")
    st.write("🌾 **Classes:** 22 Crops")
    st.write("✅ **Accuracy:** 99.32%")
    st.markdown("---")
    st.caption("B.Tech Data Science Project | 2026")

# ── Header ────────────────────────────────────────────────────
st.markdown('<div class="project-title">🌾 Smart Crop Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="project-subtitle">AgroSmart AI &nbsp;•&nbsp; Powered by Random Forest ML</div>', unsafe_allow_html=True)
st.divider()

# ── Soil Inputs ───────────────────────────────────────────────
st.markdown('<div class="section-heading">🌱 Soil Characteristics</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    N = st.number_input("Nitrogen (N) — mg/kg", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
with col2:
    P = st.number_input("Phosphorus (P) — mg/kg", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
with col3:
    K = st.number_input("Potassium (K) — mg/kg", min_value=0.0, max_value=250.0, value=50.0, step=1.0)

# ── Environmental Inputs ──────────────────────────────────────
st.markdown('<div class="section-heading">🌤️ Environmental Conditions</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
with col2:
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0, step=0.1)
with col3:
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
with col4:
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0, step=1.0)

# ── Live Metrics ──────────────────────────────────────────────
st.markdown('<div class="section-heading">📊 Current Field Conditions</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("🌱 Nitrogen",    f"{N:.1f}")
c2.metric("🌿 Phosphorus",  f"{P:.1f}")
c3.metric("🌾 Potassium",   f"{K:.1f}")
c4.metric("🧪 Soil pH",     f"{ph:.1f}")

# ── Predict ───────────────────────────────────────────────────
st.write("")
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict = st.button("🌾 RECOMMEND CROP")

if predict:
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    crop = model.predict(features)[0]

    st.markdown(f"""
    <div class="result-box">
        <div class="result-label">Recommended Crop for Your Field</div>
        <div class="result-crop">🌾 {crop.upper()}</div>
    </div>
    """, unsafe_allow_html=True)

    # Confidence breakdown
    proba = model.predict_proba(features)[0]
    top3_idx = proba.argsort()[::-1][:3]
    st.markdown('<div class="section-heading">📈 Top Crop Probabilities</div>', unsafe_allow_html=True)
    top_col1, top_col2, top_col3 = st.columns(3)
    for i, col in enumerate([top_col1, top_col2, top_col3]):
        idx = top3_idx[i]
        col.metric(
            label=f"#{i+1} {model.classes_[idx].title()}",
            value=f"{proba[idx]*100:.1f}%"
        )

# ── How It Works ──────────────────────────────────────────────
st.markdown('<div class="section-heading">⚙️ How It Works</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
col1.info("**1. Enter Field Data**\n\nProvide 7 soil & climate parameters.")
col2.info("**2. ML Analysis**\n\n100-tree Random Forest classifies across 22 crops.")
col3.info("**3. Recommendation**\n\nThe optimal crop for your conditions is returned.")

st.warning("⚠️ This system is for educational and decision-support purposes. Actual crop suitability may vary with local conditions.")

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div class="footer-text">
    🌾 AgroSmart AI &nbsp;•&nbsp; Random Forest Classifier &nbsp;•&nbsp; 99.32% Accuracy<br>
    Built with Python, scikit-learn & Streamlit
</div>
""", unsafe_allow_html=True)
