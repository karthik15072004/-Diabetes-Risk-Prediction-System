# 🩺 Diabetes Risk Prediction System

An AI-powered web application built using **Streamlit** that predicts the risk of diabetes using multiple Machine Learning algorithms. The system provides real-time predictions, model comparison, statistical analysis, personalized health recommendations, doctor suggestions, and downloadable PDF reports.

---

## 📖 Overview

Diabetes is one of the most common chronic diseases worldwide. Early detection can significantly reduce health complications and improve patient outcomes.

The **Diabetes Risk Prediction System** leverages machine learning techniques to estimate an individual's diabetes risk based on medical parameters. The application offers an interactive dashboard where users can input health data, compare different machine learning models, visualize statistical insights, and receive personalized recommendations.

---

## ✨ Features

- 🩺 Diabetes Risk Prediction
- 🤖 Multiple Machine Learning Models
- 📊 Model Performance Comparison
- 📈 ROC Curve Visualization
- 📉 Statistical Analysis Dashboard
- 📋 Dataset Summary & Descriptive Statistics
- 🔗 Correlation Heatmap
- 📦 Feature Distribution Analysis
- ⚠️ Outlier Detection
- 💡 Personalized Health Recommendations
- 👨‍⚕️ Doctor Recommendation System
- 📄 Download Prediction Report as PDF
- 📄 Download Statistical Analysis Report
- 🌙 Interactive Dark-Themed User Interface

---

## 🤖 Machine Learning Models

The project compares the performance of the following algorithms:

- Random Forest Classifier
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- XGBoost Classifier

Performance is evaluated using:

- Accuracy
- F1 Score
- ROC Curve
- Area Under Curve (AUC)

---

## 📊 Input Features

Users provide the following medical information:

- Pregnancies
- Glucose Level
- Blood Pressure
- Skin Thickness
- Insulin
- Body Mass Index (BMI)
- Diabetes Pedigree Function
- Age

---

## 📈 Statistical Analysis

The application provides:

- Descriptive Statistics
- Correlation Matrix
- Feature Distribution
- Outlier Detection
- Outcome Distribution
- High Correlation Analysis

These visualizations help users understand the dataset and the relationships among different health parameters.

---

## 👨‍⚕️ Doctor Recommendation

If the predicted diabetes risk is high, the system recommends doctors from selected cities:

- Delhi
- Chennai
- Bangalore

Each recommendation includes:

- Doctor Name
- Hospital
- Contact Number

---

## 📄 PDF Report Generation

Users can download a detailed PDF report containing:

- Patient Inputs
- Selected Machine Learning Model
- Prediction Result
- Risk Probability
- Simulated Risk
- Personalized Health Advice

---

## 🛠️ Technology Stack

### Programming Language
- Python

### Frontend
- Streamlit

### Machine Learning
- Scikit-learn
- XGBoost

### Data Processing
- Pandas
- NumPy

### Data Visualization
- Matplotlib
- Seaborn

### Report Generation
- FPDF

---

## 📂 Project Structure

```
Diabetes-Risk-Prediction-System/
│
├── app.py
├── requirements.txt
├── diabetes.csv
├── README.md
│
├── static/
├── templates/
├── reports/
└── assets/
```

---

## 🚀 Installation

### Clone the Repository

```bash
git clone https://github.com/karthik15072004/diabetes-risk-prediction-system.git
```

### Navigate to the Project

```bash
cd diabetes-risk-prediction-system
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

---

## 📊 Dataset

The project uses the **Pima Indians Diabetes Dataset**, which contains medical information collected from female patients.

Dataset Features:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome

---

## 📷 Application Modules

- 🏠 Home Page
- 📊 Prediction Dashboard
- 📈 Statistical Analysis
- 📉 Model Comparison
- 📊 ROC Curve Visualization
- 👨‍⚕️ Doctor Recommendation
- 📄 PDF Report Generator

---

## 🎯 Future Enhancements

- Deep Learning Models
- Real Hospital API Integration
- Cloud Deployment
- Electronic Health Record Integration
- User Authentication
- Database Support
- Mobile Application
- Explainable AI (XAI)

---

## 👨‍💻 Author

**Karthik Kumar**

M.Tech in Data Science  
Vellore Institute of Technology (VIT), Vellore

GitHub: https://github.com/karthik15072004

---

## ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.

---

## 📜 License

This project is intended for educational and research purposes.
