# CyberSentinel AI — Intelligent Cyber Threat Detection & Simulation Platform

A full-stack, machine learning cybersecurity web platform built with Flask, Scikit-Learn, and Vanilla CSS. The platform enables security analysts and students to ingest network flow telemetry, train supervised and unsupervised threat classifiers, investigate volumetric attacks and brute-force logins, and benchmark algorithms side-by-side with leakage-free evaluation.

---

## ⚡ Quick Start on Windows

### Method 1: One-Click Launch (Recommended)
Simply double-click:
```bat
run.bat
```
`run.bat` automatically locates the virtual environment (`.venv\Scripts\python.exe` or `..\.venv\Scripts\python.exe`), verifies all dependencies, installs missing packages if needed, and starts the server at `http://127.0.0.1:5000`.

---

### Method 2: Manual Terminal Execution (PowerShell or Command Prompt)
```powershell
# From the project root or CyberSentinel_AI directory:
..\.venv\Scripts\python.exe app.py
```

Open your browser at:
👉 **`http://127.0.0.1:5000`**

---

## 🔬 Core Features & Dashboard Capabilities

1. **Dashboard Overview Telemetry**:
   - Total Analyzed Events, Normal Traffic Count, Suspicious Traffic Count, and Detection Rate.
2. **Supported Machine Learning Models**:
   - Random Forest (Supervised Ensemble), Decision Tree (White-Box), Isolation Forest (Unsupervised Anomaly Detection).
3. **Leakage-Free Preprocessing**:
   - 25% stratified train/test split, ColumnTransformer, StandardScaler, OneHotEncoder, and median imputation.
4. **Transparent Evaluation Metrics**:
   - Real held-out metrics (Accuracy, Precision, Recall, F1) and interactive 2x2 Confusion Matrix.
5. **Threat Monitor**:
   - Live paginated event feed with search, protocol filters, status filters, and failed login warnings.
6. **Benchmark Lab**:
   - Side-by-side algorithm comparison with grouped bar charts.
7. **Secure CSV Ingestion**:
   - In-memory CSV parsing, size limit (< 8MB), format validation, and drag-and-drop support.
8. **Auxiliary Features**:
   - Dataset Preview Modal, Sample CSV Download, and Reset Demo Button.

---

## 🧪 Automated Testing

Run the test suite:
```powershell
python ..\test_suite.py
```
*(All 16 test cases verified: 100% pass rate).*
