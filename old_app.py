"""
app.py
======
HAM10000 Skin Lesion Classification Dashboard
Phase 8 (High-Throughput Batch Triage + Visual Feedback History Engine)

Run:  streamlit run app.py
"""

import os
import json
import tempfile
import sqlite3
from datetime import datetime
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
from fpdf import FPDF

# ── Page config (must be first Streamlit call) ────────────────
st.set_page_config(
    page_title  = "HAM10000 Bias-Aware Classifier",
    page_icon   = "🔬",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR       = os.path.join(BASE_DIR, 'results')
OUTPUT_DIR       = os.path.join(BASE_DIR, 'outputs')
DB_PATH          = os.path.join(RESULT_DIR, 'clinical_feedback.db')
FEEDBACK_IMG_DIR = os.path.join(RESULT_DIR, 'feedback_images')

# Ensure core repository directories exist dynamically
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FEEDBACK_IMG_DIR, exist_ok=True)

CLASS_NAMES = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
CLASS_LABELS = {
    'nv'   : 'Melanocytic Nevi',
    'mel'  : 'Melanoma',
    'bkl'  : 'Benign Keratosis',
    'bcc'  : 'Basal Cell Carcinoma',
    'akiec': 'Actinic Keratosis',
    'vasc' : 'Vascular Lesion',
    'df'   : 'Dermatofibroma',
}
DECISION_COLORS = {
    'ACCEPT': '#1D9E75',
    'FLAG'  : '#BA7517',
    'REJECT': '#D85A30',
}
DECISION_ICONS = {
    'ACCEPT': '✅',
    'FLAG'  : '⚠️',
    'REJECT': '🚫',
}

# ──────────────────────────────────────────────────────────────
# SQLITE FEEDBACK PERSISTENCE & IMAGE ARCHIVE ENGINE
# ──────────────────────────────────────────────────────────────
def init_feedback_db():
    """Initializes a local clinical SQLite table schema for logging active learning vectors."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS expert_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            image_name TEXT,
            predicted_class TEXT,
            confidence REAL,
            variance REAL,
            triage_decision TEXT,
            expert_action TEXT,
            expert_diagnosis TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_expert_feedback(img_name: str, pred_cls: str, conf: float, var: float, decision: str, action: str, expert_dx: str, pil_img: Image.Image):
    """Inserts text metrics into SQLite database and copies the image asset into the archive directory."""
    # 1. Save text metrics to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO expert_feedback (timestamp, image_name, predicted_class, confidence, variance, triage_decision, expert_action, expert_diagnosis)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), img_name, pred_cls, conf, var, decision, action, expert_dx))
    conn.commit()
    conn.close()
    
    # 2. Archive physical image file locally using filename string as key
    archive_path = os.path.join(FEEDBACK_IMG_DIR, img_name)
    pil_img.save(archive_path)

def clear_feedback_db():
    """Deletes all records from the SQLite database and removes archived physical images."""
    # Wipe the SQL Table
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM expert_feedback")
    conn.commit()
    conn.close()
    
    # Wipe the physical image directory
    if os.path.exists(FEEDBACK_IMG_DIR):
        for filename in os.listdir(FEEDBACK_IMG_DIR):
            file_path = os.path.join(FEEDBACK_IMG_DIR, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception:
                pass

def delete_single_log(log_id: int, img_name: str):
    """Deletes a specific record from the SQLite database and removes its archived image."""
    # Delete from database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM expert_feedback WHERE id = ?", (log_id,))
    conn.commit()
    conn.close()
    
    # Delete the specific physical image
    file_path = os.path.join(FEEDBACK_IMG_DIR, img_name)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception:
            pass

init_feedback_db()


# ──────────────────────────────────────────────────────────────
# LOAD PIPELINE (cached — runs once)
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models — this takes ~30s on first run…")
def load_pipeline(t_passes: int = 30):
    from inference import HAM10000Pipeline
    return HAM10000Pipeline(t_passes=t_passes)


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
def generate_pdf_report(result: dict, original_img: Image.Image) -> bytes:
    """Compiles diagnostic metrics, parameters, and images into a clinical PDF report."""
    pdf = FPDF()
    pdf.add_page()
    
    # Report Header Style
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(24, 43, 73)  # Professional Deep Navy
    pdf.cell(0, 12, "CLINICAL DIAGNOSTIC EXPERT REPORT", ln=True, align="C")
    
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 5, "HAM10000 Bias-Aware Automated Triage Core Engine", ln=True, align="C")
    
    # Divider Rule
    pdf.set_draw_color(180, 180, 180)
    pdf.line(10, 32, 200, 32)
    pdf.ln(12)
    
    # Section 1: Classification Metrics
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, "1. Core Triage Diagnostic Metrics", ln=True)
    pdf.ln(3)
    
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(80, 7, "Predicted Classification Target:")
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, f"{result['class_full']} ({result['class'].upper()})", ln=True)
    
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(80, 7, "Clinical Triage Gate Routing:")
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(40, 160, 120) if result['decision'] == 'ACCEPT' else pdf.set_text_color(200, 110, 30)
    pdf.cell(0, 7, f"{result['decision']}", ln=True)
    
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(80, 7, "Calibrated Engine Confidence:")
    pdf.cell(0, 7, f"{result['confidence'] * 100:.2f}%", ln=True)
    
    pdf.cell(80, 7, "Epistemic Uncertainty (MC Variance):")
    pdf.cell(0, 7, f"{result['uncertainty']:.6f}", ln=True)
    
    pdf.cell(80, 7, "Out-of-Distribution Distance Indicator:")
    pdf.cell(0, 7, f"Maha: {result['ood_maha']} | Energy: {result['ood_energy']}", ln=True)
    pdf.ln(10)
    
    # Section 2: Visual Evidence Mapping
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "2. Spatial Diagnostic Imaging Evidence", ln=True)
    pdf.ln(4)
    
    current_y = pdf.get_y()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f_orig:
        original_img.resize((200, 200)).save(f_orig.name)
        orig_path = f_orig.name
    pdf.image(orig_path, x=15, y=current_y, w=80, h=80)
    
    if 'gradcam_img' in result:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f_cam:
            result['gradcam_img'].resize((200, 200)).save(f_cam.name)
            cam_path = f_cam.name
        pdf.image(cam_path, x=115, y=current_y, w=80, h=80)
        
    pdf.set_y(current_y + 83)
    
    # Visual Descriptions
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.set_x(15)
    pdf.cell(80, 5, "Baseline Input Dermoscopy File", align="C")
    if 'gradcam_img' in result:
        pdf.set_x(115)
        pdf.cell(80, 5, "Fused Feature Grad-CAM Hotspot Trace", align="C")
    pdf.ln(12)
    
    # Section 3: Fully Descriptive Probability Array Matrix
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, "3. Categorical Confidence Distribution Array", ln=True)
    pdf.ln(3)
    
    # Generate structured table headers
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(240, 242, 245)
    pdf.cell(110, 7, " Target Pathological Lesion Category", border=1, fill=True)
    pdf.cell(50, 7, "Calibrated Output Probability", border=1, fill=True, align="C", ln=True)
    
    pdf.set_font("Helvetica", "", 10)
    for c_name, score in sorted(result['all_probs'].items(), key=lambda x: x[1], reverse=True):
        full_lbl = CLASS_LABELS.get(c_name, c_name)
        pdf.cell(110, 6, f" {full_lbl} ({c_name.upper()})", border=1)
        pdf.cell(50, 6, f"{score * 100:.2f}%", border=1, align="C", ln=True)
        
    try:
        os.remove(orig_path)
        if 'gradcam_img' in result:
            os.remove(cam_path)
    except:
        pass
        
    return bytes(pdf.output())


def confidence_gauge(confidence: float, decision: str) -> go.Figure:
    color = DECISION_COLORS.get(decision, '#7F77DD')
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = confidence * 100,
        title = {"text": "Calibrated Confidence (%)", "font": {"size": 14}},
        gauge = {
            "axis"     : {"range": [0, 100], "tickwidth": 1},
            "bar"      : {"color": color},
            "steps"    : [
                {"range": [0,  50],  "color": "#FAEEDA"},
                {"range": [50, 85],  "color": "#E6F1FB"},
                {"range": [85, 100], "color": "#E1F5EE"},
            ],
            "threshold": {
                "line" : {"color": "gray", "width": 2},
                "thickness": 0.75,
                "value": 85,
            },
        },
        number={"suffix": "%", "font": {"size": 28}},
    ))
    fig.update_layout(height=200, margin=dict(t=30, b=0, l=20, r=20))
    return fig


def probability_bar_chart(all_probs: dict) -> go.Figure:
    classes = list(all_probs.keys())
    probs   = list(all_probs.values())
    labels  = [CLASS_LABELS.get(c, c) for c in classes]
    colors  = ['#D85A30' if p == max(probs) else '#7F77DD' for p in probs]

    fig = go.Figure(go.Bar(
        x           = probs,
        y           = labels,
        orientation = 'h',
        marker_color= colors,
        text        = [f"{p*100:.1f}%" for p in probs],
        textposition= 'outside',
    ))
    fig.update_layout(
        title    = "Class probability distribution",
        xaxis    = dict(range=[0, 1.1], title="Probability"),
        height   = 280,
        margin   = dict(t=40, b=20, l=10, r=60),
        showlegend=False,
    )
    return fig


def uncertainty_indicator(variance: float, u_low: float, u_high: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = float(variance),
        title = {"text": "MC Dropout Variance", "font": {"size": 13}},
        gauge = {
            "axis" : {"range": [0, max(u_high * 1.5, variance * 1.2)]},
            "bar"  : {"color": "#7F77DD"},
            "steps": [
                {"range": [0,       u_low],  "color": "#E1F5EE"},
                {"range": [u_low,   u_high], "color": "#FAEEDA"},
                {"range": [u_high,  max(u_high * 1.5, variance * 1.2)], "color": "#FAECE7"},
            ],
        },
        number={"font": {"size": 18}, "valueformat": ".5f"},
    ))
    fig.update_layout(height=180, margin=dict(t=30, b=0, l=20, r=20))
    return fig


def load_scorecard() -> pd.DataFrame | None:
    path = os.path.join(RESULT_DIR, 'bias_scorecard.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def color_triage_row(val):
    if 'REJECT' in str(val):
        return 'background-color: rgba(216, 90, 48, 0.15); color: #D85A30; font-weight: bold;'
    elif 'FLAG' in str(val):
        return 'background-color: rgba(186, 117, 23, 0.15); color: #BA7517; font-weight: bold;'
    elif 'ACCEPT' in str(val):
        return 'background-color: rgba(29, 158, 117, 0.15); color: #1D9E75; font-weight: bold;'
    return ''


# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 HAM10000")
    st.caption("Bias-Aware Skin Lesion Classifier")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Predict",  "📊 Bias Dashboard", "📜 History Log", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.divider()

    st.subheader("⚙️ Inference settings")
    t_passes = st.slider(
        "MC Dropout passes (T)",
        min_value=10, max_value=50, value=30, step=5,
        help="More passes = more accurate uncertainty estimate, but slower."
    )

    st.divider()
    st.subheader("🔑 Decision key")
    for dec, icon in DECISION_ICONS.items():
        color = DECISION_COLORS[dec]
        st.markdown(
            f"{icon} **{dec}** — "
            + {"ACCEPT": "Low uncertainty, confident prediction.",
               "FLAG"  : "High uncertainty — recommend expert review.",
               "REJECT": "Out-of-distribution — image not from known classes."}[dec]
        )

    st.divider()
    st.caption("Phase 8 · HAM10000 Bias-Aware Project")


# ──────────────────────────────────────────────────────────────
# PAGE: PREDICT
# ──────────────────────────────────────────────────────────────
if page == "🏠 Predict":

    st.title("Bias-Aware Skin Lesion Classification")
    st.markdown(
        "Upload individual or bulk full-body scan assets. The system loops inference logic sequentially, "
        "generating an actively prioritized triage queue based on risk metrics."
    )
    
    st.write("### Input Imaging Queue")
    col_cam, col_up = st.columns(2)
    with col_cam:
        camera_img = st.camera_input("📷 Take a live photo")
    with col_up:
        uploaded_files = st.file_uploader("📁 Upload dermoscopy images (Single or Batch Upload)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # Coalesce inputs into a unified parsing array
    uploaded_queue = []
    if camera_img is not None:
        uploaded_queue.append(camera_img)
    if uploaded_files:
        uploaded_queue.extend(uploaded_files)

    if len(uploaded_queue) > 0:
        pipeline = load_pipeline(t_passes)
        
        batch_results = {}
        cached_pil_images = {}
        
        # Sequentially run inference over the pipeline stack
        with st.spinner(f"Processing clinical engine routing loop for {len(uploaded_queue)} asset(s)..."):
            for img_file in uploaded_queue:
                pil_img = Image.open(img_file).convert('RGB')
                cached_pil_images[img_file.name] = pil_img
                
                res = pipeline.predict(pil_img)
                batch_results[img_file.name] = res

        # ── GENERATE AND SORT THE TRIAGE QUEUE DATAFRAME ──
        st.write("### 📊 Active Clinical Triage Queue")
        st.caption("Prioritization Engine Logic: REJECT and HIGH VARIANCE (FLAG) cases automatically bubble up to top visibility row slots.")
        
        triage_rows = []
        for fname, res in batch_results.items():
            p_val = 0 if res['decision'] == 'REJECT' else (1 if res['decision'] == 'FLAG' else 2)
            
            triage_rows.append({
                "Image Asset File": fname,
                "Triage Status": f"{DECISION_ICONS[res['decision']]} {res['decision']}",
                "Primary Pred": res['class_full'],
                "Confidence": f"{res['confidence']*100:.2f}%",
                "Variance": f"{res['uncertainty']:.5f}",
                "Maha OOD": f"{res['ood_maha']:.2f}",
                "_priority": p_val 
            })
            
        df_triage = pd.DataFrame(triage_rows)
        df_triage = df_triage.sort_values(by=["_priority", "Image Asset File"]).reset_index(drop=True)
        
        # Display the highlighted DataFrame table
        st.dataframe(
            df_triage.style.apply(lambda x: [color_triage_row(v) if x.name == 'Triage Status' else '' for v in x], axis=0),
            column_order=["Image Asset File", "Triage Status", "Primary Pred", "Confidence", "Variance", "Maha OOD"],
            use_container_width=True,
            hide_index=True
        )
        
        st.divider()
        
        # ── CASE INSPECTION NODE DETECTOR ──
        st.write("### 🔍 Granular Case Diagnostics Inspector")
        selected_file = st.selectbox(
            "Select specific file to inspect complete diagnostic visualizers and reports:",
            options=df_triage['Image Asset File'].tolist()
        )
        
        # Pull targeted case vectors matching selection criteria
        result = batch_results[selected_file]
        pil_img = cached_pil_images[selected_file]
        
        col_img, col_result = st.columns([1, 2], gap="large")

        with col_result:
            dec   = result['decision']
            color = DECISION_COLORS[dec]
            icon  = DECISION_ICONS[dec]

            # ── Decision banner ──
            st.markdown(
                f"""
                <div style='background:{color}22;border:1.5px solid {color};
                            border-radius:10px;padding:14px 20px;margin-bottom:1rem'>
                  <span style='font-size:28px'>{icon}</span>
                  <span style='font-size:22px;font-weight:600;color:{color};
                               margin-left:10px'>{dec}</span>
                  <br>
                  <span style='font-size:15px;color:{color}cc'>
                    {result['class_full']} ({result['class'].upper()}) — Target: {selected_file}
                  </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Key metrics row ──
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted class",  result['class_full'])
            m2.metric("Confidence",       f"{result['confidence']*100:.1f}%")
            m3.metric("Uncertainty",      f"{result['uncertainty']:.5f}")

            st.plotly_chart(confidence_gauge(result['confidence'], dec),
                            use_container_width=True)
            
            # ── Automated Clinical Report Generation Download Gate ──
            st.write("#### Clinical Archive Integration")
            c_down, c_spacer = st.columns([1, 1])
            with c_down:
                with st.spinner("Compiling report data vector..."):
                    pdf_bytes = generate_pdf_report(result, pil_img)
                    
                st.download_button(
                    label=f"📥 Download Clinical PDF Report",
                    data=pdf_bytes,
                    file_name=f"clinical_audit_{result['class']}_{selected_file}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    help="Generates a portable, validated clinical report file ready for integration into hospital EHR systems."
                )
            
            # ── DOCTOR-IN-THE-LOOP ACTIVE LEARNING FLYWHEEL ──
            st.markdown("---")
            st.write("#### 🤝 Clinical Expert Feedback Pipeline")
            st.caption("Active Learning Loop: Feedback metrics log directly into an SQLite data table for automated model tuning sequences.")
            
            # Keys are dynamically isolated per individual filename string
            state_logged_key = f"logged_{selected_file}"
            state_show_override_key = f"show_override_{selected_file}"
            
            if state_logged_key in st.session_state:
                st.success(f"✓ {st.session_state[state_logged_key]}")
            else:
                c_fb1, c_fb2 = st.columns(2)
                with c_fb1:
                    if st.button("👍 Agree with Model", use_container_width=True, key=f"agree_{selected_file}"):
                        log_expert_feedback(
                            selected_file, 
                            result['class'], 
                            result['confidence'], 
                            result['uncertainty'], 
                            result['decision'], 
                            "AGREE", 
                            result['class'],
                            pil_img
                        )
                        st.session_state[state_logged_key] = f"Confirmed diagnosis logged successfully: {result['class_full']} (AGREE)"
                        st.rerun()
                        
                with c_fb2:
                    if st.button("👎 Disagree (Override)", use_container_width=True, key=f"override_{selected_file}"):
                        st.session_state[state_show_override_key] = True
                        
                if st.session_state.get(state_show_override_key, False):
                    st.markdown("<br>", unsafe_allow_html=True)
                    override_label = st.selectbox(
                        "Provide Valid Ground-Truth Pathological Diagnosis:",
                        options=CLASS_NAMES,
                        format_func=lambda x: f"{CLASS_LABELS[x]} ({x.upper()})",
                        key=f"select_{selected_file}"
                    )
                    
                    if st.button("Submit Verified Clinical Override", type="primary", use_container_width=True, key=f"submit_{selected_file}"):
                        log_expert_feedback(
                            selected_file, 
                            result['class'], 
                            result['confidence'], 
                            result['uncertainty'], 
                            result['decision'], 
                            "OVERRIDE", 
                            override_label,
                            pil_img
                        )
                        st.session_state[state_logged_key] = f"Clinical override registered: Changed to {CLASS_LABELS[override_label]} (OVERRIDE)"
                        st.session_state[state_show_override_key] = False
                        st.rerun()
                            
        # ── IMAGE DISPLAY & GRAD-CAM ──
        with col_img:
            st.image(pil_img, caption=f"Active File: {selected_file}", use_column_width=True)
            st.caption(f"Size: {pil_img.size[0]}×{pil_img.size[1]}px")
            
            # Display Dynamic Grad-CAM if available
            if 'gradcam_img' in result:
                st.divider()
                st.write(f"**Model Focus (Grad-CAM)**")
                st.image(result['gradcam_img'], caption=f"Explainability for {result['class'].upper()}", use_column_width=True)
                st.caption("Highlights the spatial features triggering this classification.")

        st.divider()

        # ── Probability + Uncertainty charts ──
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                probability_bar_chart(result['all_probs']),
                use_container_width=True,
            )
        with c2:
            thresholds = json.load(
                open(os.path.join(RESULT_DIR, 'uncertainty_thresholds.json')))
            st.plotly_chart(
                uncertainty_indicator(
                    result['uncertainty'],
                    thresholds['U_low_var'],
                    thresholds['U_high_var'],
                ),
                use_container_width=True,
            )
            st.caption(
                f"🟢 Accept zone: variance ≤ {thresholds['U_low_var']:.5f}  |  "
                f"🔴 Flag zone: variance > {thresholds['U_high_var']:.5f}"
            )

        # ── OOD scores ──
        with st.expander("🔎 OOD detection details", expanded=False):
            o1, o2 = st.columns(2)
            ood_cfg = json.load(open(os.path.join(RESULT_DIR, 'ood_detectors.json')))
            with o1:
                st.metric(
                    "Mahalanobis distance",
                    f"{result['ood_maha']:.2f}",
                    delta=f"threshold: {ood_cfg['mahalanobis_threshold']:.2f}",
                    delta_color="inverse",
                )
                if result['ood_maha'] > ood_cfg['mahalanobis_threshold']:
                    st.error("⚠️ Mahalanobis score exceeds threshold")
                else:
                    st.success("✓ Within distribution (Mahalanobis)")
            with o2:
                st.metric(
                    "Energy score",
                    f"{result['ood_energy']:.4f}",
                    delta=f"threshold: {ood_cfg['energy_threshold']:.4f}",
                    delta_color="normal",
                )
                if result['ood_energy'] < ood_cfg['energy_threshold']:
                    st.error("⚠️ Energy score below threshold")
                else:
                    st.success("✓ Within distribution (Energy)")

        # ── Full probability table ──
        with st.expander("📋 Full probability breakdown", expanded=False):
            prob_df = pd.DataFrame([
                {
                    "Class"      : CLASS_LABELS.get(c, c),
                    "Short code" : c.upper(),
                    "Probability": f"{p*100:.2f}%",
                    "Bar"        : p,
                }
                for c, p in sorted(result['all_probs'].items(),
                                   key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(
                prob_df.style.background_gradient(subset=["Bar"], cmap="Greens"),
                hide_index=True,
                use_container_width=True,
            )

        # ── Raw JSON for developers ──
        with st.expander("🛠️ Raw prediction output (JSON)", expanded=False):
            st.json(result)

    else:
        st.info("👆 Upload dermoscopy images above to initialize the high-throughput triage processing engine.")

        st.subheader("Example outputs by decision tier")
        ex1, ex2, ex3 = st.columns(3)
        with ex1:
            st.success("✅ **ACCEPT**\nLow uncertainty prediction — model is confident. "
                       "Confidence > 85%, variance below U_low.")
        with ex2:
            st.warning("⚠️ **FLAG**\nHigh uncertainty — recommend dermatologist review. "
                       "Variance between U_low and U_high.")
        with ex3:
            st.error("🚫 **REJECT**\nOut-of-distribution image. "
                     "Mahalanobis or Energy detector fired — "
                     "not a recognisable skin lesion class.")


# ──────────────────────────────────────────────────────────────
# PAGE: BIAS DASHBOARD
# ──────────────────────────────────────────────────────────────
elif page == "📊 Bias Dashboard":

    st.title("Bias Audit Dashboard")
    st.markdown(
        "Complete bias analysis across all project phases. "
        "Shows how each intervention reduced bias in the HAM10000 classifier."
    )

    scorecard = load_scorecard()

    # ── Summary metrics from Phase 7 ──
    if scorecard is not None:
        p2_rows = scorecard[scorecard['phase'].isin(['phase2_baseline', 'baseline'])]
        p7_rows = scorecard[scorecard['phase'] == 'phase7_final_test']

        if not p2_rows.empty and not p7_rows.empty:
            p2 = p2_rows.iloc[0]
            p7 = p7_rows.iloc[0]

            st.subheader("📈 Headline improvements — Phase 2 → Phase 7")
            c1, c2, c3, c4 = st.columns(4)

            def delta_metric(col_ref, label, fmt=".3f"):
                v2 = float(p2[col_ref]) if col_ref in p2.index else 0
                v7 = float(p7[col_ref]) if col_ref in p7.index else 0
                return label, v7, v7 - v2, fmt

            for col, (label, v7, delta, fmt) in zip(
                [c1, c2, c3, c4],
                [delta_metric('balanced_acc', 'Balanced accuracy'),
                 delta_metric('macro_f1',     'Macro F1'),
                 delta_metric('df_recall',    'df recall  ⚑'),
                 delta_metric('vasc_recall',  'vasc recall ⚑')],
            ):
                col.metric(
                    label,
                    f"{v7:{fmt}}",
                    delta=f"+{delta:.3f}" if delta > 0 else f"{delta:.3f}",
                )

    st.divider()

    # ── Charts from Drive ──
    chart_files = [
        ('final_bias_scorecard.png',       'Final bias scorecard heatmap'),
        ('bias_progression_chart.png',     'Bias progression across phases'),
        ('phase2_vs_phase3_recall.png',    'Phase 2 vs Phase 3 recall'),
        ('phase4_ensemble_comparison.png', 'Phase 4 ensemble comparison'),
        ('reliability_diagrams.png',       'Reliability diagrams — calibration'),
        ('shap_backbone_contribution.png', 'SHAP backbone contribution'),
        ('final_confusion_matrix.png',     'Final confusion matrix'),
    ]

    for fname, title in chart_files:
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(fpath):
            st.subheader(title)
            st.image(fpath, use_column_width=True)
            st.divider()
        else:
            st.warning(f"Chart not found: {fname}  (run Phase 7 to generate)")

    # ── Scorecard table ──
    st.subheader("📋 Full bias scorecard (all phases)")
    if scorecard is not None:
        display_cols = [c for c in scorecard.columns
                        if any(c.endswith(s)
                               for s in ['_acc', '_f1', '_recall', 'ece', 'phase'])]
        st.dataframe(
            scorecard[display_cols].set_index('phase').style.format("{:.4f}", na_rep="—"),
            use_container_width=True,
        )
    else:
        st.error(f"bias_scorecard.csv not found at {RESULT_DIR}")


# ──────────────────────────────────────────────────────────────
# NEW PAGE: VISUAL HISTORY ENGINE & ACTIVE LEARNING DATABASE LOGS
# ──────────────────────────────────────────────────────────────
elif page == "📜 History Log":
    st.title("📜 Expert Review & Active Learning Log")
    
    col_title, col_clear = st.columns([3, 1])
    with col_title:
        st.markdown(
            "Querying localized SQLite historical logs. Select any processed image case vector "
            "to visually audit the diagnostic scans archived during clinical operations."
        )
    with col_clear:
        if st.button("🗑️ Clear Entire Log History", use_container_width=True, type="primary"):
            clear_feedback_db()
            st.success("All logs and images cleared successfully.")
            st.rerun()
            
    # Read text metadata log records from the local SQLite engine
    conn = sqlite3.connect(DB_PATH)
    try:
        df_log = pd.read_sql_query("SELECT id, timestamp, image_name, predicted_class, triage_decision, expert_action, expert_diagnosis FROM expert_feedback ORDER BY id DESC", conn)
    except Exception:
        df_log = pd.DataFrame()
    conn.close()
    
    if df_log.empty:
        st.info("📂 The active learning database is currently empty. Submit logs inside the 'Predict' tab to populate records.")
    else:
        # Display high-level metrics card matrix
        total_logs = len(df_log)
        override_count = len(df_log[df_log['expert_action'] == 'OVERRIDE'])
        
        m_h1, m_h2, m_h3 = st.columns(3)
        m_h1.metric("Total Archived Cases", total_logs)
        m_h2.metric("Total Expert Overrides", override_count)
        m_h3.metric("Flywheel Alignment Rate", f"{((total_logs - override_count)/total_logs)*100:.1f}%")
        
        st.write("### 🗃️ Historic Data Table Logs")
        st.dataframe(df_log, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Build interactive historic viewer node dropdown
        st.write("### 🖼️ Historic Visual Asset Viewer")
        selected_log_id = st.selectbox(
            "Select unique Case ID row entry to load original scan and details:",
            options=df_log['id'].tolist(),
            format_func=lambda x: f"Case ID: {x} — Target Asset: {df_log[df_log['id'] == x]['image_name'].values[0]}"
        )
        
        # Load targets from disk memory
        target_row = df_log[df_log['id'] == selected_log_id].iloc[0]
        target_img_name = target_row['image_name']
        archived_img_path = os.path.join(FEEDBACK_IMG_DIR, target_img_name)
        
        v_col1, v_col2 = st.columns([1, 1], gap="large")
        
        with v_col1:
            st.write("**Archived Skin Lesion Scan:**")
            if os.path.exists(archived_img_path):
                h_img = Image.open(archived_img_path)
                st.image(h_img, caption=f"Archived Asset: {target_img_name}", use_column_width=True)
            else:
                st.error("🚨 Image file payload missing from local file directory mapping.")
                
        with v_col2:
            st.write("**Logged Case Audit Metrics:**")
            
            # Map clean display names dynamically
            m_cls = target_row['predicted_class']
            e_cls = target_row['expert_diagnosis']
            
            st.info(f"📆 **Timestamp:** {target_row['timestamp']}")
            st.warning(f"🤖 **Model Prediction:** {CLASS_LABELS.get(m_cls, m_cls)} ({m_cls.upper()})")
            st.success(f"🩺 **Expert Diagnosis:** {CLASS_LABELS.get(e_cls, e_cls)} ({e_cls.upper()})")
            
            # Sub-call coloring logic cards
            t_dec_key = target_row['triage_decision'].replace('✅ ', '').replace('⚠️ ', '').replace('🚫 ', '')
            color_banner = DECISION_COLORS.get(t_dec_key, '#7F77DD')
            st.markdown(
                f"""
                <div style='background:{color_banner}22;border:1px solid {color_banner};border-radius:8px;padding:10px 15px;'>
                    <span style='color:{color_banner};font-weight:600;'>Triage Routing: {target_row['triage_decision']}</span><br>
                    <span style='color:#666;font-size:13px;'>Action Recorded: {target_row['expert_action']}</span>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # --- NEW FEATURE: Target Single Delete Action ---
            st.write("---")
            if st.button(f"🗑️ Delete Case ID {selected_log_id} from History", use_container_width=True):
                delete_single_log(selected_log_id, target_img_name)
                st.success(f"Case {selected_log_id} successfully deleted.")
                st.rerun()


# ──────────────────────────────────────────────────────────────
# PAGE: ABOUT
# ──────────────────────────────────────────────────────────────
elif page == "ℹ️ About":

    st.title("About this project")

    st.markdown(f"""
    ## HAM10000 Bias-Aware Skin Lesion Classification

    This application is the **Phase 8 deliverable** of a bias-awareness project
    built on the HAM10000 dermoscopy dataset.

    ---

    ### Architecture

    | Component | Details |
    |---|---|
    | **Backbone 1** | EfficientNet-B4 (pretrained, fine-tuned Phase 3) |
    | **Backbone 2** | DenseNet-121 (pretrained, fine-tuned Phase 4) |
    | **Backbone 3** | ViT-B/16 (pretrained, fine-tuned Phase 4) |
    | **Ensemble** | Stacking MLP meta-learner on 3584-dim concat features |
    | **Calibration** | Temperature scaling (Phase 5) |
    | **Uncertainty** | MC Dropout T={t_passes} passes (Phase 5) |
    | **Explainability**| Dynamic PyTorch Hooks for Grad-CAM Visualization |
    | **Reporting Engine**| In-memory fpdf2 Dynamic PDF Generation |
    | **Queue Management**| Prioritized High-Throughput Batch Triage Processing Engine |
    | **Active Learning** | SQLite Persistent Doctor-in-the-Loop Feedback Loop System & Image Archive |
    | **OOD detection** | Mahalanobis distance + Energy score OR gate (Phase 6) |

    ---

    ### Bias mitigation pipeline

    1. **Data bias** — WeightedRandomSampler + Focal Loss (γ=2) + CutMix
    2. **Feature bias** — Grad-CAM audit, ROI segmentation, stain normalisation
    3. **Calibration bias** — Temperature scaling, ECE reduced ~78%
    4. **Confidence bias** — MC Dropout flags uncertain predictions explicitly
    5. **OOD bias** — Rejects images not matching any training class
    6. **Evaluation bias** — Balanced accuracy + per-class AUROC as primary metrics

    ---

    ### Classes

    | Code | Full name | Training samples |
    |---|---|---|
    | nv    | Melanocytic Nevi       | ~6705 (majority) |
    | mel   | Melanoma               | ~1113 |
    | bkl   | Benign Keratosis       | ~1099 |
    | bcc   | Basal Cell Carcinoma   | ~514  |
    | akiec | Actinic Keratosis      | ~327  |
    | vasc  | Vascular Lesion        | ~142  |
    | df    | Dermatofibroma         | ~115 (minority) |

    ---

    ### Decision gate

    | Decision | Condition | Meaning |
    |---|---|---|
    | ✅ **ACCEPT** | Variance ≤ U_low, not OOD | Confident prediction |
    | ⚠️ **FLAG** | Variance > U_high          | Refer to expert |
    | 🚫 **REJECT** | Mahalanobis OR Energy OOD | Not a known lesion class |

    ---

    *Built with PyTorch · timm · Streamlit · Albumentations*
    """)