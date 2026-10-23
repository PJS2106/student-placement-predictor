import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Student Placement Predictor", page_icon="🎓", layout="wide")

# Demo dataset for a classroom prototype only.
# This synthetic data is generated to demonstrate the workflow, not to reproduce the
# internship dataset or claim validated real-world accuracy.
@st.cache_resource
def train_demo_model():
    rng = np.random.default_rng(42)
    n = 400
    cgpa = rng.uniform(5.0, 10.0, n).round(2)
    attendance = rng.integers(50, 101, n)
    projects = rng.integers(0, 7, n)
    internships = rng.integers(0, 4, n)
    coding = rng.integers(30, 101, n)
    communication = rng.integers(30, 101, n)
    aptitude = rng.integers(30, 101, n)

    # Simulated target for demo purposes only.
    score = (cgpa - 5) * 7 + (attendance - 50) * 0.12 + projects * 2.5 + internships * 4 \
            + coding * 0.12 + communication * 0.06 + aptitude * 0.10
    probability = 1 / (1 + np.exp(-(score - 52) / 9))
    placed = rng.binomial(1, probability)
    data = pd.DataFrame({
        "CGPA": cgpa,
        "Attendance": attendance,
        "Projects": projects,
        "Internships": internships,
        "Coding_Skills": coding,
        "Communication": communication,
        "Aptitude": aptitude,
        "Placed": placed
    })
    X = data.drop(columns="Placed")
    y = data["Placed"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    return model, accuracy, cm, data

model, demo_accuracy, demo_cm, demo_data = train_demo_model()

st.title("🎓 Student Placement Prediction System")
st.caption("Data Science Internship Prototype | Zavico AI Solutions Co. | Student: Sridharshan PJ")
st.info("Demonstration prototype: because the original placement.csv and trained model were not provided, this app uses a synthetic sample dataset. Predictions and metrics are illustrative, not validated real-world results.")

tab_predict, tab_dashboard, tab_about = st.tabs(["🔮 Predict Placement", "📊 Dataset & Model", "ℹ️ About"])

with tab_predict:
    st.subheader("Enter Student Details")
    st.write("Fill in the student's academic and skill details, then run the model.")
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=8.4, step=0.1)
            attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=92)
            projects = st.number_input("Number of Projects", min_value=0, max_value=20, value=4)
        with c2:
            internships = st.number_input("Internships", min_value=0, max_value=10, value=2)
            coding = st.slider("Coding Skills (0–100)", 0, 100, 88)
            communication = st.slider("Communication (0–100)", 0, 100, 85)
        with c3:
            aptitude = st.slider("Aptitude (0–100)", 0, 100, 75)
            st.write("")
            st.write("")
            submitted = st.form_submit_button("Predict Placement", type="primary", use_container_width=True)

    if submitted:
        features = pd.DataFrame([{
            "CGPA": cgpa,
            "Attendance": attendance,
            "Projects": projects,
            "Internships": internships,
            "Coding_Skills": coding,
            "Communication": communication,
            "Aptitude": aptitude
        }])
        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
        placed_probability = float(probabilities[list(model.classes_).index(1)])
        st.divider()
        st.subheader("Prediction Result")
        left, right = st.columns([1, 2])
        with left:
            if prediction == 1:
                st.success("Likely to be Placed")
            else:
                st.warning("Not Likely to be Placed")
            st.metric("Estimated placement probability", f"{placed_probability * 100:.1f}%")
        with right:
            st.progress(placed_probability, text="Model-estimated placement probability")
            st.write("This is a prediction from the demonstration model, not a guarantee of actual placement.")

with tab_dashboard:
    st.subheader("Demo Dataset Overview")
    m1, m2, m3 = st.columns(3)
    m1.metric("Synthetic records", len(demo_data))
    m2.metric("Input features", 7)
    m3.metric("Demo test accuracy", f"{demo_accuracy * 100:.1f}%")
    st.caption("Accuracy is calculated on the generated demo dataset only. It must not be reported as the accuracy from the internship report.")
    st.write("Preview of the sample dataset")
    st.dataframe(demo_data.head(10), use_container_width=True)
    st.write("Placement distribution")
    counts = demo_data["Placed"].map({0: "Not Placed", 1: "Placed"}).value_counts()
    st.bar_chart(counts)
    st.write("Confusion matrix (demo model)")
    cm_df = pd.DataFrame(demo_cm, index=["Actual Not Placed", "Actual Placed"],
                         columns=["Predicted Not Placed", "Predicted Placed"])
    st.dataframe(cm_df, use_container_width=True)

with tab_about:
    st.subheader("Project Workflow")
    st.markdown("""
    1. Dataset preparation
    2. Data preprocessing
    3. Exploratory Data Analysis (EDA)
    4. Feature-based Random Forest training
    5. Model evaluation
    6. Student placement prediction
    """)
    st.subheader("Technologies")
    st.markdown("Python · Pandas · NumPy · Scikit-learn · Streamlit")
    st.caption("Created as a review prototype based on the project described in the internship report. Replace the synthetic demo data/model with the original dataset and trained model when available.")
