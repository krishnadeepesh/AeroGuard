"""
AeroGuard - Air Quality Category Prediction & Grounded Health Advisory
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="AeroGuard | Air Quality & Health Advisory",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SRC_DIR = os.path.join(BASE_DIR, "src")

import sys
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from data_loader import FEATURE_COLUMNS, AQI_CLASSES, calculate_sub_index, BP_PM25, BP_PM10, BP_NO2, BP_SO2, BP_CO, BP_O3
from rag_engine import RAGEngine

# Clean styling
st.markdown("""
<style>
    .badge-good { background-color: #28a745; color: white; padding: 4px 12px; border-radius: 12px; font-weight: bold; }
    .badge-satisfactory { background-color: #5cb85c; color: white; padding: 4px 12px; border-radius: 12px; font-weight: bold; }
    .badge-moderate { background-color: #ffc107; color: black; padding: 4px 12px; border-radius: 12px; font-weight: bold; }
    .badge-poor { background-color: #fd7e14; color: white; padding: 4px 12px; border-radius: 12px; font-weight: bold; }
    .badge-very-poor { background-color: #dc3545; color: white; padding: 4px 12px; border-radius: 12px; font-weight: bold; }
    .badge-severe { background-color: #6f42c1; color: white; padding: 4px 12px; border-radius: 12px; font-weight: bold; }
    .rag-quote {
        background-color: #f8f9fa;
        border-left: 4px solid #0d6efd;
        padding: 10px 14px;
        margin-top: 8px;
        border-radius: 4px;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_ml_artifacts():
    rf_model = joblib.load(os.path.join(MODELS_DIR, "aqi_random_forest.joblib"))
    baseline_model = joblib.load(os.path.join(MODELS_DIR, "aqi_logistic_regression.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    with open(os.path.join(MODELS_DIR, "evaluation_results.json"), "r") as f:
        eval_data = json.load(f)
    return rf_model, baseline_model, scaler, eval_data

@st.cache_resource
def get_rag_engine():
    return RAGEngine(use_ollama=True)

try:
    rf_clf, baseline_clf, scaler, eval_results = load_ml_artifacts()
    rag_engine = get_rag_engine()
except Exception as e:
    st.error(f"Error loading system components: {e}")
    st.stop()

def get_badge_html(category):
    class_map = {
        "Good": "badge-good",
        "Satisfactory": "badge-satisfactory",
        "Moderate": "badge-moderate",
        "Poor": "badge-poor",
        "Very Poor": "badge-very-poor",
        "Severe": "badge-severe"
    }
    css_cls = class_map.get(category, "badge-moderate")
    return f'<span class="{css_cls}">{category.upper()}</span>'

# Sidebar Navigation
st.sidebar.title("🌫️ AeroGuard")
app_mode = st.sidebar.radio(
    "Menu",
    [
        "Air Quality & Health Advisory",
        "Model Analytics",
        "Test Scenarios"
    ]
)

# ========================================================
# VIEW 1: WORKING APPLICATION (PREDICTION + ADVISORY)
# ========================================================
if app_mode == "Air Quality & Health Advisory":
    st.title("Air Quality Prediction & Health Advisory")
    st.markdown("Adjust atmospheric measurements to predict air quality and generate health guidance.")

    # Quick Presets
    col_preset1, col_preset2, col_preset3, col_preset4 = st.columns(4)
    preset_choice = None
    if col_preset1.button("🌿 Clean Day"):
        preset_choice = "clean"
    if col_preset2.button("🏙️ Moderate Urban"):
        preset_choice = "moderate"
    if col_preset3.button("⚠️ Poor / Winter"):
        preset_choice = "winter"
    if col_preset4.button("🚨 Severe Smog"):
        preset_choice = "severe"

    presets = {
        "clean": {"PM2.5": 14.0, "PM10": 28.0, "NO2": 18.0, "SO2": 8.0, "CO": 0.5, "O3": 25.0, "Temperature": 27.5, "Humidity": 82.0},
        "moderate": {"PM2.5": 72.0, "PM10": 165.0, "NO2": 68.0, "SO2": 24.0, "CO": 1.7, "O3": 55.0, "Temperature": 28.0, "Humidity": 58.0},
        "winter": {"PM2.5": 195.0, "PM10": 395.0, "NO2": 210.0, "SO2": 42.0, "CO": 4.6, "O3": 90.0, "Temperature": 13.5, "Humidity": 78.0},
        "severe": {"PM2.5": 330.0, "PM10": 520.0, "NO2": 310.0, "SO2": 58.0, "CO": 7.2, "O3": 118.0, "Temperature": 11.0, "Humidity": 86.0}
    }

    current_vals = presets.get(preset_choice, presets["moderate"])

    st.subheader("Sensor Measurements")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pm25 = st.slider("PM2.5 (ug/m³)", 0.0, 500.0, float(current_vals["PM2.5"]), 0.5)
        pm10 = st.slider("PM10 (ug/m³)", 0.0, 600.0, float(current_vals["PM10"]), 1.0)
    with col2:
        no2 = st.slider("NO2 (ug/m³)", 0.0, 450.0, float(current_vals["NO2"]), 1.0)
        so2 = st.slider("SO2 (ug/m³)", 0.0, 300.0, float(current_vals["SO2"]), 1.0)
    with col3:
        co = st.slider("CO (mg/m³)", 0.0, 20.0, float(current_vals["CO"]), 0.1)
        o3 = st.slider("O3 (ug/m³)", 0.0, 350.0, float(current_vals["O3"]), 1.0)
    with col4:
        temp = st.slider("Temperature (°C)", -5.0, 50.0, float(current_vals["Temperature"]), 0.5)
        humidity = st.slider("Humidity (%)", 10.0, 100.0, float(current_vals["Humidity"]), 0.5)

    input_dict = {
        "PM2.5": pm25, "PM10": pm10, "NO2": no2, "SO2": so2,
        "CO": co, "O3": o3, "Temperature": temp, "Humidity": humidity
    }

    # Calculate sub-indices to pinpoint dominant pollutant
    sub_indices = {
        "PM2.5": calculate_sub_index(pm25, BP_PM25),
        "PM10": calculate_sub_index(pm10, BP_PM10),
        "NO2": calculate_sub_index(no2, BP_NO2),
        "SO2": calculate_sub_index(so2, BP_SO2),
        "CO": calculate_sub_index(co, BP_CO),
        "O3": calculate_sub_index(o3, BP_O3)
    }
    dominant_pollutant = max(sub_indices, key=sub_indices.get)
    max_subindex = sub_indices[dominant_pollutant]

    # ML Inference
    input_df = pd.DataFrame([input_dict])[FEATURE_COLUMNS]
    input_scaled = scaler.transform(input_df)
    rf_pred = rf_clf.predict(input_scaled)[0]
    rf_proba = rf_clf.predict_proba(input_scaled)[0]
    rf_confidence = float(np.max(rf_proba))

    st.markdown("---")
    st.subheader("Air Quality Classification")

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.markdown("#### AQI Category")
        st.markdown(f"<h2>{get_badge_html(rf_pred)}</h2>", unsafe_allow_html=True)
        st.markdown(f"**Confidence:** `{rf_confidence * 100:.1f}%`")

    with res_col2:
        st.markdown("#### Primary Pollutant")
        st.markdown(f"### 🎯 **{dominant_pollutant}**")
        st.markdown(f"Calculated Sub-Index: **{max_subindex:.1f}**")

    # Class probability distribution bar
    prob_df = pd.DataFrame({
        "AQI Class": rf_clf.classes_,
        "Probability": rf_proba
    }).sort_values(by="Probability", ascending=False)
    st.bar_chart(prob_df.set_index("AQI Class"))

    # Advisory Section
    st.markdown("---")
    st.subheader("Health Advisory & Recommendations")

    ml_packet = {
        "predicted_bucket": rf_pred,
        "confidence": rf_confidence,
        "dominant_pollutant": dominant_pollutant
    }

    if st.button("Generate Health Advisory", type="primary"):
        with st.spinner("Generating health advisory based on current readings..."):
            rag_output = rag_engine.generate_advisory(ml_packet, input_dict)

        advisory_col, source_col = st.columns([2, 1])

        with advisory_col:
            st.markdown("### Public Advisory")
            st.markdown(rag_output["advisory"])

        with source_col:
            st.markdown("### Retrieved Reference Guidelines")
            for src in rag_output.get("retrieved_sources", []):
                st.markdown(f"""
                <div class="rag-quote">
                    <b>Source:</b> {src.get('source', '')}<br>
                    <b>Section:</b> {src.get('section', '')}<br>
                    <p style="margin-top:6px; color:#333;">{src.get('text', '')[:200]}...</p>
                </div>
                """, unsafe_allow_html=True)

# ========================================================
# VIEW 2: MODEL ANALYTICS
# ========================================================
elif app_mode == "Model Analytics":
    st.title("Model Performance")

    rf_res = eval_results["primary_random_forest"]
    base_res = eval_results["baseline_logistic_regression"]

    # Comparative Metrics Table
    metrics_summary = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score"
        ],
        "Logistic Regression (Baseline)": [
            f"{base_res['accuracy']*100:.2f}%",
            f"{base_res['report']['macro avg']['precision']*100:.2f}%",
            f"{base_res['report']['macro avg']['recall']*100:.2f}%",
            f"{base_res['macro_f1']*100:.2f}%"
        ],
        "Random Forest": [
            f"{rf_res['accuracy']*100:.2f}%",
            f"{rf_res['macro_precision']*100:.2f}%",
            f"{rf_res['macro_recall']*100:.2f}%",
            f"{rf_res['macro_f1']*100:.2f}%"
        ]
    })
    st.table(metrics_summary)

    col_cm1, col_cm2 = st.columns(2)
    with col_cm1:
        st.markdown("**Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            rf_res["confusion_matrix"],
            annot=True, fmt="d", cmap="Greens",
            xticklabels=AQI_CLASSES, yticklabels=AQI_CLASSES, ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

    with col_cm2:
        st.markdown("**Feature Importance**")
        feat_df = pd.DataFrame(
            list(rf_res["feature_importances"].items()),
            columns=["Feature", "Importance"]
        ).sort_values(by="Importance", ascending=False)
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        sns.barplot(x="Importance", y="Feature", data=feat_df, palette="viridis", ax=ax2)
        st.pyplot(fig2)

# ========================================================
# VIEW 3: TEST SCENARIOS
# ========================================================
elif app_mode == "Test Scenarios":
    st.title("Test Scenarios")

    test_cases = [
        {
            "name": "Scenario 1: Clean Coastal Day",
            "desc": "Typical clean atmospheric condition.",
            "inputs": {"PM2.5": 12.0, "PM10": 24.0, "NO2": 15.0, "SO2": 8.0, "CO": 0.4, "O3": 20.0, "Temperature": 26.0, "Humidity": 85.0},
            "expected": "Good"
        },
        {
            "name": "Scenario 2: Winter Smog Episode",
            "desc": "Heavy particulate stagnation during winter inversion.",
            "inputs": {"PM2.5": 320.0, "PM10": 510.0, "NO2": 320.0, "SO2": 65.0, "CO": 7.5, "O3": 120.0, "Temperature": 9.5, "Humidity": 90.0},
            "expected": "Severe"
        },
        {
            "name": "Scenario 3: Borderline Satisfactory / Moderate",
            "desc": "Concentration near threshold boundary.",
            "inputs": {"PM2.5": 61.0, "PM10": 98.0, "NO2": 45.0, "SO2": 18.0, "CO": 1.2, "O3": 48.0, "Temperature": 24.0, "Humidity": 55.0},
            "expected": "Moderate"
        }
    ]

    selected_test = st.selectbox("Select Scenario:", [c["name"] for c in test_cases])
    test_data = next(c for c in test_cases if c["name"] == selected_test)

    st.markdown(f"**Description:** {test_data['desc']}")

    col_t1, col_t2 = st.columns([1, 1])
    with col_t1:
        st.json(test_data["inputs"])

    with col_t2:
        test_df = pd.DataFrame([test_data["inputs"]])[FEATURE_COLUMNS]
        test_scaled = scaler.transform(test_df)
        pred = rf_clf.predict(test_scaled)[0]
        prob = float(np.max(rf_clf.predict_proba(test_scaled)[0]))

        st.markdown(f"### Predicted: {get_badge_html(pred)}", unsafe_allow_html=True)
        st.markdown(f"**Expected:** `{test_data['expected']}`")
        st.markdown(f"**Confidence:** `{prob*100:.1f}%`")
        if pred == test_data["expected"]:
            st.success("Result matches expected category.")
