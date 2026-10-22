# CyberSentinel AI — Intelligent Cyber Threat Detection & Simulation Platform

A full-stack, machine learning cybersecurity web platform built with Flask, Scikit-Learn, and Vanilla CSS. The platform enables security analysts and students to ingest network flow telemetry, train supervised and unsupervised threat classifiers, investigate volumetric attacks and brute-force logins, and benchmark algorithms side-by-side with leakage-free evaluation.

---

## ⚡ Quick Start on Windows

### Method 1: One-Click Launch (Recommended)
Simply double-click:
```bat
run.bat
```
`run.bat` automatically locates the virtual environment (`.venv\Scripts\python.exe`), verifies all dependencies, installs missing packages if needed, and starts the server at `http://127.0.0.1:5000`.

---

### Method 2: Manual Terminal Execution (PowerShell or Command Prompt)
PowerShell often restricts script execution policies for `Activate.ps1`. You do **not** need to change system-wide security settings. Run Python directly from `.venv`:

```powershell
# 1. Verify dependencies are installed
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Run the application
.\.venv\Scripts\python.exe app.py
```

Open your browser at:
👉 **`http://127.0.0.1:5000`**

---

## 🔬 Core Features & Dashboard Capabilities

1. **Dashboard Overview Telemetry**:
   - **Total Analyzed Events**: Total network flow connections ingested.
   - **Normal Traffic Count**: Verified benign communication flows.
   - **Suspicious Traffic Count**: High-risk or anomalous connections flagged.
   - **Detection Rate**: Percentage of flows flagged by the active model.

2. **Supported Machine Learning Models**:
   - **Random Forest**: Supervised ensemble of 100 decorrelated decision trees with balanced class weighting. Provides high precision and robust threat detection.
   - **Decision Tree**: Supervised white-box CART model with Gini impurity splitting; fully interpretable classification thresholds.
   - **Isolation Forest**: Unsupervised anomaly detection. Isolates outliers in feature space without requiring prior ground-truth labels during training. Anomaly predictions (`-1`) are mapped to suspicious flags.

3. **Leakage-Free Preprocessing Pipeline**:
   - Stratified train/test split (25% held-out test set) isolates test records *before* fitting transformers.
   - `ColumnTransformer` with `SimpleImputer` (median/most-frequent), `StandardScaler` for numeric features, and `OneHotEncoder` for categorical protocols.

4. **Transparent Evaluation Metrics**:
   - Real held-out metrics calculated on unseen test data: **Accuracy**, **Precision**, **Recall**, and **F1 Score**.
   - Interactive 2x2 **Confusion Matrix**: True Normal (TN), False Positive (FP), False Negative (FN), and True Positive (TP).

5. **Threat Monitor (Live Event Feed)**:
   - Paginated network flow table (10 items per page with Previous / Next navigation).
   - Real-time search across Event ID, Protocol, and Packet counts.
   - Filter by Protocol (`TCP`, `UDP`, `HTTP`, `DNS`) or Status (`Normal`, `Suspicious`).
   - Clear visual status badges and highlighted indicators for repeat failed logins.

6. **Interactive Benchmark Lab**:
   - Side-by-side comparison of Decision Tree, Random Forest, and Isolation Forest on identical held-out test splits.
   - Comparative grouped bar chart and benchmark metrics table.

7. **Secure Data Ingestion & Drag-and-Drop**:
   - Accepts `.csv` files up to 8 MB with in-memory parsing (no disk execution or path traversal risk).
   - Real-time file validation feedback and "Clear file" button to return to the sample dataset.

8. **Auxiliary Features**:
   - **Preview Dataset Modal**: Inspect the first 10 rows of the active dataset inline.
   - **Download Sample CSV**: Export the built-in synthetic network flow records.
   - **Reset Demo Button**: Instantly restores default parameters and re-analyzes baseline data.

---

## 📁 Repository Structure

```text
CyberSentinel_AI_Project/
├── .venv/                         # Python virtual environment
├── app.py                         # Root launcher application
├── requirements.txt               # Project dependencies
├── run.bat                        # Windows 1-click startup batch script
├── test_suite.py                  # Automated test verification suite
├── README.md                      # Documentation
└── CyberSentinel_AI/
    ├── app.py                     # Main Flask backend & ML engine
    ├── requirements.txt           # Package requirements
    ├── run.bat                    # Local directory run script
    ├── setup.bat                  # Setup script
    ├── data/
    │   ├── network_traffic.csv    # Synthetic network flows (800 rows)
    │   └── README.txt             # Data documentation
    ├── templates/
    │   └── index.html             # Dashboard UI template
    └── static/
        ├── style.css              # Dark cybersecurity styling
        └── app.js                 # Frontend interactions & Chart.js logic
```

---

## 📊 Dataset Schema

The system ingests network connection flows with the following features:

| Column | Type | Description |
|---|---|---|
| `duration` | Float | Session/flow duration in seconds |
| `protocol` | String | Transport/application protocol (TCP, UDP, HTTP, DNS) |
| `packets` | Integer | Total packet count in the flow |
| `bytes` | Integer | Total bytes transferred |
| `failed_logins` | Integer | Count of failed authentication attempts |
| `label` | Integer | Ground truth label (0 = Normal, 1 = Suspicious) |

*Flexible Aliasing Supported*: `connection_duration`, `total_packets`, `total_bytes`, `failed_attempts`, `attack`, `class`, `target`.

---

## 🧪 Automated Testing

Execute the automated test suite verifying all 16 backend integration cases:

```powershell
.\.venv\Scripts\python.exe test_suite.py
```

### Verified Test Cases:
1. `GET /` — Homepage loads with 200 OK.
2. `POST /api/analyze` — Random Forest default execution and metric extraction.
3. `POST /api/analyze` — Decision Tree execution.
4. `POST /api/analyze` — Isolation Forest with custom contamination slider.
5. `POST /api/analyze` — In-memory custom CSV upload parsing.
6. `POST /api/analyze` — Rejection of invalid non-CSV file extensions.
7. `POST /api/analyze` — Rejection of CSVs missing the ground-truth `label` column.
8. `POST /api/analyze` — Rejection of insufficient rows (< 10 rows).
9. `POST /api/compare` — Benchmark lab across all three algorithms.
10. `GET /api/sample` — Sample CSV download attachment.
11. `GET /api/preview` — Raw sample records preview endpoint.
12. `make_sample_data()` — Reproducible synthetic data generation.

---

## ⚠️ Important Educational Limitations

- **Educational Prototype**: CyberSentinel AI is an educational demonstration and research simulation platform.
- **Synthetic Data**: The built-in dataset is synthetic and demonstrates known volumetric spikes and brute-force patterns. Metrics must not be represented as proof of real-world intrusion defense against zero-day exploits.
- **Production IDS Requirements**: Real-world intrusion detection systems require raw packet capture (PCAP), deep packet inspection (DPI), encrypted traffic analysis (JA3/TLS fingerprinting), and continuous model calibration.
