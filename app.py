import streamlit as st
import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback

from skimage.feature import hog
from PIL import Image

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Hand Gesture Recognition",
    page_icon="✋",
    layout="wide"
)

# ==================================================
# BUILT-IN STYLING
# ==================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

h1, h2, h3 {
    text-align: center;
}

.stMetric {
    border: 1px solid #444;
    border-radius: 10px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD MODEL
# ==================================================

try:
    model = joblib.load("models/gesture_svm.pkl")
    scaler = joblib.load("models/scaler.pkl")

except Exception:
    st.error("❌ Model Loading Failed")
    st.code(traceback.format_exc())
    st.stop()

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("✋ Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "Prediction",
        "About Project"
    ]
)

# ==================================================
# DASHBOARD
# ==================================================

if page == "Dashboard":

    st.title("✋ AI Hand Gesture Recognition System")

    st.markdown("""
    <div style="
    background:#00e5ff;
    color:black;
    padding:12px;
    border-radius:10px;
    text-align:center;
    font-weight:bold;
    margin-bottom:20px;">
    SkillCraft Technology ML Internship - Hand Gesture Recognition System
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "### HOG + SVM Based Gesture Classification"
    )

    try:

        banner = Image.open(
            "hand_gesture_recognition.jpg"
        )

        c1, c2, c3 = st.columns([1,2,1])

        with c2:
            st.image(
                banner,
                width=650
            )

    except:
        st.warning(
            "Place HandGestureRecognition.jpg in project folder."
        )

    st.markdown("---")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Model", "SVM")
    m2.metric("Feature", "HOG")
    m3.metric("Classes", "10")
    m4.metric("Output", "Gesture")

    st.markdown("---")

    left, right = st.columns(2)

    with left:

        st.subheader("📋 Project Overview")

        st.info("""
        • Image Upload

        • Image Preprocessing

        • HOG Feature Extraction

        • Feature Scaling

        • SVM Classification

        • Gesture Prediction
        """)

    with right:

        st.subheader("🛠 Technology Stack")

        st.success("""
        • Python

        • OpenCV

        • Scikit-Learn

        • HOG

        • SVM

        • Streamlit
        """)

    st.markdown("---")

    st.subheader("🎯 Objective")

    st.write("""
    Develop a machine learning model capable of
    recognizing and classifying hand gestures
    from image data.
    """)

# ==================================================
# PREDICTION PAGE
# ==================================================

elif page == "Prediction":

    st.title("✋ Gesture Prediction")

    uploaded_file = st.file_uploader(
        "📤 Upload Hand Gesture Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        file_bytes = np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            rgb,
            caption="Uploaded Gesture",
            use_container_width=True
        )

        # ---------------------------
        # PREPROCESSING
        # ---------------------------

        img = cv2.resize(
            image,
            (128, 128)
        )

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        features = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            feature_vector=True
        )

        features = scaler.transform(
            [features]
        )

        # ---------------------------
        # PREDICTION
        # ---------------------------

        prediction = model.predict(
            features
        )[0]

        probabilities = model.predict_proba(
            features
        )[0]

        confidence = (
            np.max(probabilities) * 100
        )

        st.markdown("---")

        st.subheader("Prediction Result")

        st.success(
            f"Detected Gesture: {prediction}"
        )

        st.metric(
            "Confidence Score",
            f"{confidence:.2f}%"
        )

        st.progress(
            float(confidence / 100)
        )

        st.markdown("---")

        st.subheader(
            "Gesture Probability Distribution"
        )

        classes = model.classes_

        prob_df = pd.DataFrame({
            "Gesture": classes,
            "Probability": probabilities * 100
        })

        fig, ax = plt.subplots(
            figsize=(8, 4)
        )

        ax.bar(
            prob_df["Gesture"],
            prob_df["Probability"]
        )

        ax.set_ylabel(
            "Probability (%)"
        )

        ax.set_title(
            "Prediction Confidence"
        )

        plt.xticks(rotation=45)

        plt.tight_layout()

        st.pyplot(fig)

# ==================================================
# ABOUT PAGE
# ==================================================

else:

    st.title(
        "📚 Project Information"
    )

    st.subheader(
        "Project Objective"
    )

    st.write("""
    Build a machine learning model capable of
    recognizing and classifying hand gestures.
    """)

    st.subheader(
        "Dataset"
    )

    st.write("""
    LeapGestRecog Dataset

    Classes:

    • Palm
    • L
    • Fist
    • Fist Moved
    • Thumb
    • Index
    • OK
    • Palm Moved
    • C
    • Down
    """)

    st.subheader(
        "Machine Learning Pipeline"
    )

    pipeline = pd.DataFrame({
        "Step": [
            "Image Collection",
            "Image Resize",
            "Gray Conversion",
            "HOG Features",
            "Scaling",
            "SVM Training",
            "Prediction"
        ]
    })

    st.table(pipeline)

    st.subheader(
        "Future Enhancements"
    )

    st.write("""
    • Real-Time Webcam Detection

    • MediaPipe Hand Tracking

    • CNN-Based Models

    • Gesture-Based Controls

    • Cloud Deployment
    """)

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.caption(
    "Developed by Gopi Krishna | SkillCraft Technology ML Internship"
)