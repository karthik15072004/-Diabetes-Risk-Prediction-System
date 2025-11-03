import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
from fpdf import FPDF
import xgboost as xgb

# -----------------------------------------------------------
# Streamlit Page Config & Custom Dark Theme
# -----------------------------------------------------------
st.set_page_config(page_title="🩺 Diabetes Risk Prediction System", layout="wide")

st.markdown("""
<style>
body {
    background: radial-gradient(circle at top left, #0d1b2a, #1b263b, #0d1b2a);
    color: #e0e1dd;
    font-family: 'Segoe UI', sans-serif;
}
h1, h2, h3, h4 {
    color: #00e6ac;
    text-align: center;
    font-weight: 700;
}
.stButton > button {
    background: linear-gradient(90deg, #00bfa5, #00796b);
    color: white;
    border-radius: 12px;
    padding: 0.8em;
    font-size: 17px;
    width: 100%;
    border: none;
    transition: all 0.3s ease-in-out;
    box-shadow: 0px 0px 10px rgba(0,255,200,0.4);
}
.stButton > button:hover {
    background: linear-gradient(90deg, #00e6ac, #00bfa5);
    transform: scale(1.05);
    box-shadow: 0px 0px 20px rgba(0,255,200,0.7);
}
.stContainer {
    background: rgba(255, 255, 255, 0.08);
    padding: 2.5rem;
    border-radius: 20px;
    box-shadow: 0 4px 30px rgba(0,0,0,0.6);
    margin: 2rem auto;
    width: 85%;
    backdrop-filter: blur(8px);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# Navigation
# -----------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_home():
    st.session_state.page = "home"

def go_dashboard():
    st.session_state.page = "dashboard"


# -----------------------------------------------------------
# Load Model and Dataset
# -----------------------------------------------------------
@st.cache_resource
def train_all_models():
    df = pd.read_csv(r"C:\Users\karth\Downloads\diabetes.csv")
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True, random_state=42),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    accuracy_results = {}
    f1_results = {}
    roc_data = {}
    trained_models = {}

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)

        accuracy_results[name] = acc
        f1_results[name] = f1
        roc_data[name] = (fpr, tpr, roc_auc)
        trained_models[name] = clf

    # Combine results
    acc_df = pd.DataFrame({
        "Model": list(accuracy_results.keys()),
        "Accuracy (%)": [v * 100 for v in accuracy_results.values()],
        "F1 Score": list(f1_results.values())
    }).sort_values(by="Accuracy (%)", ascending=False)

    return trained_models, scaler, X.columns, acc_df, roc_data


trained_models, scaler, feature_names, acc_df, roc_data = train_all_models()


# -----------------------------------------------------------
# DOCTOR LIST
# -----------------------------------------------------------
doctors = {
    "Delhi": [
        {"name": "Dr. A Sharma", "hospital": "Delhi Diabetes Center", "phone": "+91 98765 43210"},
        {"name": "Dr. B Verma", "hospital": "Fortis Hospital", "phone": "+91 98765 54321"},
        {"name": "Dr. C Gupta", "hospital": "Max Hospital", "phone": "+91 98765 67890"},
        {"name": "Dr. D Singh", "hospital": "BLK Super Speciality", "phone": "+91 98765 11122"},
        {"name": "Dr. E Khanna", "hospital": "Indraprastha Apollo", "phone": "+91 98765 33344"}
    ],
    "Chennai": [
        {"name": "Dr. P Rajan", "hospital": "Apollo Sugar Clinic", "phone": "+91 98450 12345"},
        {"name": "Dr. R Natarajan", "hospital": "MIOT Hospital", "phone": "+91 90940 56789"},
        {"name": "Dr. S Balaji", "hospital": "SRM Hospital", "phone": "+91 90940 67890"},
        {"name": "Dr. T Krishnan", "hospital": "Global Hospitals", "phone": "+91 90940 78901"},
        {"name": "Dr. Q Srinivasan", "hospital": "Fortis Malar Hospital", "phone": "+91 90940 45678"}
    ],
    "Bangalore": [
        {"name": "Dr. K Reddy", "hospital": "Bangalore Diabetes Center", "phone": "+91 99887 66554"},
        {"name": "Dr. L Kumar", "hospital": "Manipal Hospital", "phone": "+91 99887 77665"},
        {"name": "Dr. M Iyer", "hospital": "Columbia Asia", "phone": "+91 99887 88776"},
        {"name": "Dr. N Shetty", "hospital": "Sakra World Hospital", "phone": "+91 99887 99887"},
        {"name": "Dr. O Das", "hospital": "Fortis Hospital", "phone": "+91 99887 11223"}
    ]
}

# -----------------------------------------------------------
# HOME PAGE
# -----------------------------------------------------------
if st.session_state.page == "home":
    st.markdown("<h1>🩺 Diabetes Risk Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center; color:#b0bec5;'>Your AI-Powered Multi-Model Health Assistant</h4>", unsafe_allow_html=True)
    st.markdown("""
    <div class='stContainer'>
    <h3>✨ Welcome!</h3>
    <p>This AI-driven dashboard predicts <b>Diabetes Risk</b> using multiple machine learning models — Random Forest, Logistic Regression, KNN, SVM, and XGBoost.</p>
    <ul>
        <li>⚙️ Instant predictions with 5 algorithms</li>
        <li>📊 Compare model accuracies and ROC curves</li>
        <li>🧠 Personalized health advice</li>
        <li>👩‍⚕️ Doctor recommendations near you</li>
        <li>📄 Downloadable PDF reports</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        if st.button("🚀 Launch Prediction Dashboard"):
            go_dashboard()
    st.markdown("<hr><div style='text-align:center;color:gray;'>👨‍💻 Developed by Karthik</div>", unsafe_allow_html=True)

# -----------------------------------------------------------
# DASHBOARD PAGE
# -----------------------------------------------------------
elif st.session_state.page == "dashboard":
    st.markdown("<h1>📊 Diabetes Prediction Dashboard</h1>", unsafe_allow_html=True)

    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose Level (mg/dL)", 0, 300, 120)
    blood_pressure = st.number_input("Blood Pressure (mmHg)", 0, 200, 70)
    skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
    insulin = st.number_input("Insulin Level (µU/mL)", 0, 900, 80)
    bmi = st.number_input("BMI (kg/m²)", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.number_input("Age", 1, 120, 30)

    selected_model_name = st.selectbox("🧠 Choose Model", list(trained_models.keys()))
    selected_model = trained_models[selected_model_name]

    if st.button("🔍 Predict Diabetes Risk"):
        data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]]
        scaled = scaler.transform(data)
        prediction = selected_model.predict(scaled)[0]
        probability = selected_model.predict_proba(scaled)[0][1]
        st.session_state.prediction = prediction
        st.session_state.probability = probability
        st.session_state.model_name = selected_model_name

    if "prediction" in st.session_state:
        pred = st.session_state.prediction
        prob = st.session_state.probability
        model_name = st.session_state.model_name

        st.metric(f"{model_name} Probability", f"{prob*100:.2f}%")
        if pred == 1:
            st.error("⚠️ High Risk of Diabetes")
        else:
            st.success("✅ Low Risk of Diabetes")

        st.subheader("💡 Personalized Advice")
        tips = [
            "Maintain healthy BMI and regular exercise.",
            "Eat high-fiber, low-sugar foods.",
            "Regular medical checkups are important.",
            "Avoid smoking and alcohol."
        ]
        for tip in tips:
            st.write(f"- {tip}")

        if pred == 1:
            city = st.selectbox("Select City for Doctor Recommendations", list(doctors.keys()))
            if st.button("Find Doctors"):
                for doc in doctors[city]:
                    st.markdown(f"""
                    <div class='stContainer'>
                    <b>👨‍⚕️ {doc['name']}</b><br>
                    🏥 {doc['hospital']}<br>
                    📞 {doc['phone']}
                    </div>
                    """, unsafe_allow_html=True)

        # Real-time simulation
        st.subheader("⚡ Real-time Simulation")
        sim_glucose = st.slider("Glucose Level", 0, 300, glucose)
        sim_bmi = st.slider("BMI", 0.0, 70.0, bmi)
        sim_insulin = st.slider("Insulin", 0, 900, insulin)
        sim_data = [[pregnancies, sim_glucose, blood_pressure, skin_thickness, sim_insulin, sim_bmi, dpf, age]]
        sim_scaled = scaler.transform(sim_data)
        sim_prob = selected_model.predict_proba(sim_scaled)[0][1]
        st.info(f"🩸 Simulated Risk Probability: **{sim_prob*100:.2f}%**")

        # -------------------------------
        # 📊 Model Comparison Table
        # -------------------------------
        st.subheader("📊 Model Accuracy & F1 Score Comparison")
        st.dataframe(acc_df)

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x="Accuracy (%)", y="Model", data=acc_df, ax=ax, palette="crest")
        for i, (acc, model) in enumerate(zip(acc_df["Accuracy (%)"], acc_df["Model"])):
            ax.text(acc + 0.3, i, f"{acc:.2f}%", va='center')
        ax.set_xlabel("Accuracy (%)")
        ax.set_title("Model Performance Comparison")
        st.pyplot(fig)

        # -------------------------------
        # 🩸 ROC Curves
        # -------------------------------
        st.subheader("🩸 ROC Curves for All Models")

        fig, ax = plt.subplots(figsize=(7, 6))
        for model_name, (fpr, tpr, roc_auc) in roc_data.items():
            ax.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.2f})")

        ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves Comparison")
        ax.legend(loc="lower right")
        st.pyplot(fig)

        # -------------------------------
        # 📄 PDF Report
        # -------------------------------
        def create_pdf(inputs, pred, prob, sim_prob, model_name):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 12, "Diabetes Prediction Report", ln=True, align="C")
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, f"Model Used: {model_name}", ln=True)
            pdf.ln(5)
            for k, v in inputs.items():
                pdf.cell(0, 8, f"{k}: {v}", ln=True)
            pdf.ln(5)
            pdf.cell(0, 10, f"Predicted Risk: {'High' if pred == 1 else 'Low'}", ln=True)
            pdf.cell(0, 10, f"Probability: {prob*100:.2f}%", ln=True)
            pdf.cell(0, 10, f"Simulated Risk: {sim_prob*100:.2f}%", ln=True)
            pdf.ln(5)
            pdf.cell(0, 10, "Advice:", ln=True)
            for t in tips:
                pdf.multi_cell(0, 8, f"- {t}")
            return pdf.output(dest='S').encode('latin-1')

        pdf_bytes = create_pdf({
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "Blood Pressure": blood_pressure,
            "Skin Thickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DPF": dpf,
            "Age": age
        }, pred, prob, sim_prob, model_name)

        st.download_button("📄 Download PDF Report", pdf_bytes, "Diabetes_Report.pdf", "application/pdf")

    if st.button("🏠 Back to Home"):
        go_home()
# -----------------------------------------------------------
# STATISTICAL ANALYSIS PAGE (compact version)
# -----------------------------------------------------------
def go_statistics():
    st.session_state.page = "statistics"

# Navigation button in Dashboard
if st.session_state.page == "dashboard":
    if st.button("📈 Go to Statistical Analysis"):
        go_statistics()

# -----------------------------------------------------------
# 📊 STATISTICAL ANALYSIS PAGE CONTENT
# -----------------------------------------------------------
elif st.session_state.page == "statistics":
    st.markdown("<h1>📊 Statistical Analysis of Dataset</h1>", unsafe_allow_html=True)
    df = pd.read_csv(r"C:\Users\karth\Downloads\diabetes.csv")

    st.markdown("""
    <div class='stContainer'>
    <h3>📘 Overview</h3>
    <p>This section gives a concise statistical overview of the Diabetes dataset, showing key metrics,
    feature relationships, and data distributions that influence model predictions.</p>
    </div>
    """, unsafe_allow_html=True)

    # -----------------------------
    # Descriptive Statistics
    # -----------------------------
    st.markdown("### 🧮 Descriptive Statistics")
    styled_df = (
        df.describe()
        .T
        .style.background_gradient(cmap="viridis")
        .format(precision=2)
    )
    st.dataframe(styled_df, use_container_width=True)

    # -----------------------------
    # Correlation Heatmap (smaller)
    # -----------------------------
    st.markdown("### 🔗 Correlation Heatmap")
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.4, square=True, ax=ax1)
    ax1.set_title("Feature Correlation Matrix", fontsize=11, pad=6)
    st.pyplot(fig1)

    # -----------------------------
    # Highly Correlated Features
    # -----------------------------
    st.markdown("### 💡 Highly Correlated Feature Pairs (r > 0.7)")
    corr_matrix = df.corr().abs()
    high_corr_pairs = (
        corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )
    high_corr_pairs.columns = ["Feature 1", "Feature 2", "Correlation"]
    high_corr_pairs = high_corr_pairs[high_corr_pairs["Correlation"] > 0.7]
    if not high_corr_pairs.empty:
        st.dataframe(high_corr_pairs, use_container_width=True)
    else:
        st.info("No highly correlated features found (r > 0.7).")

    # -----------------------------
    # Feature Distribution (smaller)
    # -----------------------------
    st.markdown("### 📊 Feature Distribution")
    feature = st.selectbox("Select a feature to view distribution", df.columns)
    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    sns.histplot(df[feature], bins=20, kde=True, ax=ax2, color="#00bfa5")
    ax2.set_title(f"Distribution of {feature}", fontsize=11, pad=6)
    st.pyplot(fig2)

    # -----------------------------
    # Outlier Detection (compact)
    # -----------------------------
    st.markdown("### ⚠️ Outlier Detection (Box Plot)")
    feature_box = st.selectbox("Select feature for box plot", df.columns, key="boxplot")
    fig3, ax3 = plt.subplots(figsize=(5, 3))
    sns.boxplot(x=df[feature_box], color="#00e6ac", ax=ax3)
    ax3.set_title(f"Outlier Detection for {feature_box}", fontsize=11, pad=6)
    st.pyplot(fig3)

    # -----------------------------
    # Outcome Distribution (smallest)
    # -----------------------------
    st.markdown("### 🧠 Outcome Count (Target Variable)")
    fig4, ax4 = plt.subplots(figsize=(4, 3))
    sns.countplot(x="Outcome", data=df, palette="crest", ax=ax4)
    ax4.set_title("Outcome Distribution", fontsize=11, pad=6)
    st.pyplot(fig4)

    # -----------------------------
    # Summary Insights
    # -----------------------------
    st.markdown("<h3>🧠 Key Insights</h3>", unsafe_allow_html=True)
    insights = [
        "Higher glucose and BMI values are strong diabetes indicators.",
        "Insulin and Skin Thickness show high variance with outliers.",
        "Data scaling helps improve model learning consistency.",
        "Outcome variable is fairly balanced across samples."
    ]
    for ins in insights:
        st.markdown(f"- {ins}")

    # -----------------------------
    # PDF Report
    # -----------------------------
    from fpdf import FPDF
    import tempfile, os

    class PDF(FPDF):
        def header(self):
            self.set_font("Arial", "B", 13)
            self.cell(0, 10, "Diabetes Dataset Statistical Summary", ln=True, align="C")
            self.ln(4)
        def section_title(self, title):
            self.set_font("Arial", "B", 11)
            self.cell(0, 8, title, ln=True)
            self.ln(1)
        def section_body(self, text):
            self.set_font("Arial", "", 9)
            self.multi_cell(0, 7, text)
            self.ln(3)

    def create_statistics_pdf(df, insights):
        pdf = PDF()
        pdf.add_page()

        pdf.section_title("Descriptive Statistics Summary:")
        stats_summary = df.describe().round(2).to_string()
        pdf.section_body(stats_summary)

        pdf.section_title("Highly Correlated Features (r > 0.7):")
        corr_matrix = df.corr().abs()
        high_corr_pairs = (
            corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            .stack()
            .reset_index()
        )
        high_corr_pairs.columns = ["Feature 1", "Feature 2", "Correlation"]
        high_corr_pairs = high_corr_pairs[high_corr_pairs["Correlation"] > 0.7]
        if not high_corr_pairs.empty:
            pdf.section_body(high_corr_pairs.to_string(index=False))
        else:
            pdf.section_body("No highly correlated features found (r > 0.7).")

        pdf.section_title("Insights:")
        for ins in insights:
            pdf.cell(0, 7, f"- {ins}", ln=True)

        pdf.ln(5)
        pdf.set_font("Arial", "I", 8)
        pdf.cell(0, 8, "Generated by Diabetes Prediction System", ln=True, align="C")

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf.output(tmp.name)
        return tmp.name

    if st.button("📄 Download Statistics Report as PDF"):
        pdf_path = create_statistics_pdf(df, insights)
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="⬇️ Click to Download Report",
                data=f,
                file_name="Diabetes_Statistics_Report.pdf",
                mime="application/pdf"
            )
        os.remove(pdf_path)

    # -----------------------------
    # Navigation Buttons
    # -----------------------------
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Dashboard"):
            go_dashboard()
    with col2:
        if st.button("🏠 Back to Home"):
            go_home()
