import streamlit as st
import tensorflow as tf
import urllib.request
import numpy as np
import zipfile
import json
import os
from PIL import Image

# ==========================================
# 🎨 STYLING & PALETTE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Agri-Extract GAN Diagnostics Dashboard",
    page_icon="🌾",
    layout="wide"
)

# Custom injection for agricultural domain themes
st.markdown("""
    <style>
    .main { background-color: #f7f9f6; }
    h1, h2, h3 { color: #1e4620 !important; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .report-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #2e7d32;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .metric-box {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #c8e6c9;
    }
    .model-winner {
        border: 2px solid #2e7d32 !important;
        background-color: #f1f8e9 !important;
    }
    </style>
""", unsafe_allow_html=True)

CATEGORIES = [
    "Bacterial Leaf Blight", "Brown Spot", "Healthy Rice Leaf",
    "Leaf Blast", "Leaf scald", "Sheath Blight"
]

MIGRATION_DIR = "./metrics_checkpoint_payload"

# ==========================================
# 📂 CACHED SYSTEM LOADING UTILITIES
# ==========================================

# Direct Hugging Face download paths mapped to your exact filenames
P1_URL = "https://huggingface.co/atomdev-ibktommy/rice-leaf-pathology-models/resolve/main/pipeline_1_baseline.keras"
P2_URL = "https://huggingface.co/atomdev-ibktommy/rice-leaf-pathology-models/resolve/main/pipeline_2_hybrid.keras"
ZIP_URL = "https://huggingface.co/atomdev-ibktommy/rice-leaf-pathology-models/resolve/main/streamlit_test_samples.zip"

P1_LOCAL_PATH = "pipeline_1_baseline.keras"
P2_LOCAL_PATH = "pipeline_2_hybrid.keras"
ZIP_LOCAL_PATH = "streamlit_test_samples.zip"
EXTRACTED_DIR = "./streamlit_test_samples"  # Extracted to match your folder preference
MANIFEST_LOCAL_PATH = "test_samples_manifest.json"


@st.cache_resource
def load_diagnostic_models():
    """Dynamically stream network weights into the server instance container."""
    if not os.path.exists(P1_LOCAL_PATH):
        with st.spinner("Downloading Baseline Model from Hugging Face..."):
            urllib.request.urlretrieve(P1_URL, P1_LOCAL_PATH)

    if not os.path.exists(P2_LOCAL_PATH):
        with st.spinner("Downloading Hybrid GAN Model from Hugging Face..."):
            urllib.request.urlretrieve(P2_URL, P2_LOCAL_PATH)

    try:
        p1 = tf.keras.models.load_model(P1_LOCAL_PATH, compile=False)
        p2 = tf.keras.models.load_model(P2_LOCAL_PATH, compile=False)
        return p1, p2
    except Exception as e:
        st.error(f"❌ Error compiling model architectures: {e}")
        return None, None


@st.cache_data
def extract_and_parse_test_pool():
    """Download and extract test images using our deterministic manifest lookup schema."""
    # Step 1: Download and unpack the zip archive if it's missing locally
    if not os.path.exists(EXTRACTED_DIR):
        if not os.path.exists(ZIP_LOCAL_PATH):
            with st.spinner("Streaming preset validation testing pool from remote archive..."):
                urllib.request.urlretrieve(ZIP_URL, ZIP_LOCAL_PATH)

        with st.spinner("Unpacking research specimens for interactive matrix evaluation..."):
            with zipfile.ZipFile(ZIP_LOCAL_PATH, 'r') as zip_ref:
                zip_ref.extractall(".")  # Extracts directly as 'streamlit_test_samples' folder
            try:
                os.remove(ZIP_LOCAL_PATH)
            except Exception:
                pass

    # Step 2: Correlate directories directly using the local tracking manifest
    indexed_samples = []
    if os.path.exists(MANIFEST_LOCAL_PATH):
        with open(MANIFEST_LOCAL_PATH, 'r') as f:
            manifest_data = json.load(f)

        for file_name, meta in manifest_data.items():
            # Standardize root folder structures dynamically
            clean_rel_path = meta["relative_path"].replace("./streamlit_test_samples/", "")
            full_disk_path = os.path.join(EXTRACTED_DIR, clean_rel_path)

            if os.path.exists(full_disk_path):
                indexed_samples.append({
                    "path": full_disk_path,
                    "name": file_name,
                    "true_label": meta["true_label"]
                })

    # Fallback to structural walking if manifest is unreadable
    if not indexed_samples and os.path.exists(EXTRACTED_DIR):
        for root, dirs, files in os.walk(EXTRACTED_DIR):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(root, file)
                    matched_lbl = "Unverified"
                    for cat in CATEGORIES:
                        if cat in root or cat in file:
                            matched_lbl = cat
                    indexed_samples.append({
                        "path": full_path, "name": file, "true_label": matched_lbl
                    })

    return indexed_samples


# Initialize networks
model_p1, model_p2 = load_diagnostic_models()
test_pool = extract_and_parse_test_pool()


# ==========================================
# 📊 PREDICTION ENGINE HANDLER
# ==========================================
def run_model_inference(model, image_pil):
    try:
        input_shape = model.input_shape
        target_h, target_w = input_shape[1], input_shape[2]
    except Exception:
        target_h, target_w = 180, 180

    img_rgb = image_pil.convert("RGB")
    img_resized = img_rgb.resize((target_w, target_h))
    img_array = np.array(img_resized, dtype=np.float32)

    img_tensor = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_tensor, verbose=0)[0]
    best_idx = np.argmax(predictions)

    return CATEGORIES[best_idx], predictions[best_idx], predictions


# ==========================================
# 🖥️ CORE UI RENDERING FRAMEWORK
# ==========================================
st.title("🌾 Agri-Extract: Deep Pathology & GAN Evaluation Dashboard")
st.markdown(
    "Welcome back! Use this decision support platform to analyze crop conditions across both standard architectures.")

# Setup sidebar documentation controls
st.sidebar.header("🚜 Navigation & Parameters")
app_mode = st.sidebar.radio(
    "Select Diagnostic Workspace:",
    ["📊 Comparative Evaluation Lab", "🔬 Historical Performance Analytics"]
)

comp_data_path = os.path.join(MIGRATION_DIR, "cross_pipeline_performance_manifest.json")
has_history = os.path.exists(comp_data_path)

# ------------------------------------------
# WORKSPACE 1: COMPARATIVE INFERENCE LAB
# ------------------------------------------
if app_mode == "📊 Comparative Evaluation Lab":
    st.subheader("🌾 Dual-Model Inference Analysis Platform")

    src_toggle = st.radio("Choose Input Feed Source:", [
        "🎯 Select Preset Research Test Pool",
        "📸 Upload New Field Inspection Photo"
    ], horizontal=True)

    active_image = None
    expected_ground_truth = None

    if src_toggle == "🎯 Select Preset Research Test Pool":
        if not test_pool:
            st.warning(
                "⚠️ Local lookup assets initializing or missing. Re-verify payload configurations.")
        else:
            sample_options = [f"{i + 1}: {s['name']} [{s['true_label']}]" for i, s in
                              enumerate(test_pool)]
            selected_idx = st.selectbox("Select Target Plant Sample from Test Pool:",
                                        range(len(sample_options)),
                                        format_func=lambda x: sample_options[x])

            chosen_sample = test_pool[selected_idx]
            active_image = Image.open(chosen_sample["path"]).convert("RGB")
            expected_ground_truth = chosen_sample["true_label"]

    else:
        uploaded_file = st.file_uploader("Drop field leaf photo here (JPEG/PNG):",
                                         type=["jpg", "jpeg", "png"])
        if uploaded_file:
            active_image = Image.open(uploaded_file).convert("RGB")
            expected_ground_truth = st.selectbox(
                "Specify True Ground-Truth Condition (If Known/Confirmed by Agrologist):",
                ["Unverified / Not Labeled"] + CATEGORIES
            )
            if expected_ground_truth == "Unverified / Not Labeled":
                expected_ground_truth = None

    if active_image is not None:
        st.markdown("---")
        col_img, col_metrics = st.columns([1, 2])

        with col_img:
            st.image(active_image, caption="Active Diagnostic Specimen", use_column_width=True)
            if expected_ground_truth:
                st.info(f"📋 Verified Ground Truth Label: **{expected_ground_truth}**")
            else:
                st.warning("📋 Ground Truth Status: **Unlabeled Custom Sample**")

        with col_metrics:
            st.markdown("### 🧠 Live Cross-Network Predictions")

            if model_p1 is None or model_p2 is None:
                st.error("Model architectures are uninitialized. Check weight dependencies.")
            else:
                lbl_p1, conf_p1, all_p1 = run_model_inference(model_p1, active_image)
                lbl_p2, conf_p2, all_p2 = run_model_inference(model_p2, active_image)

                p1_matches = (lbl_p1 == expected_ground_truth) if expected_ground_truth else False
                p2_matches = (lbl_p2 == expected_ground_truth) if expected_ground_truth else False

                c1, c2 = st.columns(2)

                with c1:
                    is_winner = (conf_p1 > conf_p2) and (p1_matches or not expected_ground_truth)
                    div_class = "report-card model-winner" if is_winner else "report-card"

                    st.markdown(f"""
                        <div class="{div_class}">
                            <h4>Pipeline 1: Traditional Augmentation</h4>
                            <hr style='margin: 8px 0;'>
                            <p>Predicted Diagnosis:<br><b><span style='color:#c27d0e; font-size:17px;'>{lbl_p1}</span></b></p>
                            <p>Confidence Level: <b>{conf_p1 * 100:.2f}%</b></p>
                        </div>
                    """, unsafe_allow_html=True)

                    if expected_ground_truth:
                        if p1_matches:
                            st.success("🎯 Correct Prediction Match!")
                        else:
                            st.error("❌ Prediction Mismatch")

                with c2:
                    is_winner = (conf_p2 > conf_p1) and (p2_matches or not expected_ground_truth)
                    div_class = "report-card model-winner" if is_winner else "report-card"

                    st.markdown(f"""
                        <div class="{div_class}">
                            <h4>Pipeline 2: Targeted cGAN Hybrid</h4>
                            <hr style='margin: 8px 0;'>
                            <p>Predicted Diagnosis:<br><b><span style='color:#2e7d32; font-size:17px;'>{lbl_p2}</span></b></p>
                            <p>Confidence Level: <b>{conf_p2 * 100:.2f}%</b></p>
                        </div>
                    """, unsafe_allow_html=True)

                    if expected_ground_truth:
                        if p2_matches:
                            st.success("🎯 Correct Prediction Match!")
                        else:
                            st.error("❌ Prediction Mismatch")

                st.markdown("### 🏛️ Ascertainment Evaluation Report")

                if expected_ground_truth:
                    if p1_matches and p2_matches:
                        higher_model = "Pipeline 2 (cGAN Hybrid)" if conf_p2 > conf_p1 else "Pipeline 1 (Traditional)"
                        margin = abs(conf_p2 - conf_p1) * 100
                        st.success(
                            f"🤝 **Both models accurately matched the true label.** **{higher_model}** is selected as the optimal deployment choice due to a higher diagnostic confidence margin of **+{margin:.2f}%**.")
                    elif p2_matches:
                        st.success(
                            "🚀 **Pipeline 2 (cGAN Hybrid Model) successfully hit the true diagnosis while the baseline failed.** The targeted synthetic balancing resolved minority feature limits!")
                    elif p1_matches:
                        st.warning(
                            "⚠️ **Pipeline 1 (Traditional Baseline) successfully matched the true label while the hybrid model missed.** Review sample distribution properties.")
                    else:
                        st.error(
                            "💥 **Neither network successfully identified this pathology specimen.** The sample features fall outside current model boundaries.")
                else:
                    higher_model = "Pipeline 2 (cGAN Hybrid)" if conf_p2 > conf_p1 else "Pipeline 1 (Traditional)"
                    margin = abs(conf_p2 - conf_p1) * 100
                    st.info(
                        f"🔍 **Unlabeled Sample Diagnostic Output:** **{higher_model}** provides the highest diagnostic certainty profile for this specimen, outperforming the alternate configuration by a margin of **{margin:.2f}%** confidence.")

# ------------------------------------------
# WORKSPACE 2: HISTORICAL METRICS ANALYTICS
# ------------------------------------------
else:
    st.subheader("🔬 Thesis Experimental Analytics Vault")
    st.markdown(
        "Review the macro comparative benchmarks generated during active research runs below.")

    if not has_history:
        st.warning(
            "⚠️ Cross-pipeline comparison logs manifest missing from payload path directory.")
    else:
        with open(comp_data_path, 'r') as f:
            c_manifest = json.load(f)

        p1_acc = c_manifest["pipeline_1_accuracy"]
        p2_acc = c_manifest["pipeline_2_accuracy"]
        lift = (p2_acc - p1_acc) * 100

        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""
                <div class="metric-box">
                    <h5>Pipeline 1 Test Accuracy</h5>
                    <h2>{p1_acc * 100:.2f}%</h2>
                    <p style='color:#777;'>Traditional Uniform Augmentation</p>
                </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
                <div class="metric-box" style='border-color:#81c784;'>
                    <h5>Pipeline 2 Test Accuracy</h5>
                    <h2>{p2_acc * 100:.2f}%</h2>
                    <p style='color:#2e7d32; font-weight:bold;'>Targeted cGAN Balanced (+{lift:.2f}% Lift)</p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📈 Visualizing Static Checkpoint Assets")

        col_curve, col_cm = st.columns(2)
        with col_curve:
            img_c_path = os.path.join(MIGRATION_DIR, 'pipeline2_learning_curves.png')
            if os.path.exists(img_c_path):
                with open(img_c_path, "rb") as f:
                    st.image(f.read(), caption="Pipeline 2 Loss & Accuracy Optimization Paths",
                             use_column_width=True)
            else:
                st.info("Learning curves plot artifact missing from disk payload directory.")

        with col_cm:
            img_m_path = os.path.join(MIGRATION_DIR, 'pipeline_cross_comparison_f1.png')
            if os.path.exists(img_m_path):
                with open(img_m_path, "rb") as f:
                    st.image(f.read(), caption="Per-Class Cross-Pipeline F1 Evolution Bar Chart",
                             use_column_width=True)
            else:
                st.info(
                    "F1 comparative grouped bar chart plot missing from disk payload directory.")