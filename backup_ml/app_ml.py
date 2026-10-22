"""
CyberSentinel AI — Intelligent Cyber Threat Detection and Simulation Platform
Backend Flask Application & Machine Learning Engine

Educational threat-detection prototype demonstrating supervised classification
(Random Forest, Decision Tree) and unsupervised anomaly detection (Isolation Forest)
on network traffic flows.
"""

import os
import io
import sys
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Determine paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "network_traffic.csv")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(DATA_DIR, exist_ok=True)

# Initialize Flask application with explicit folder references
app = Flask(
    __name__,
    template_folder=TEMPLATES_DIR,
    static_folder=STATIC_DIR
)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload ceiling

# Core expected network traffic feature schema
FEATURES = ["duration", "protocol", "packets", "bytes", "failed_logins"]
TARGET = "label"


def make_sample_data(n=800, seed=42):
    """
    Generate a reproducible synthetic network-traffic dataset.
    NOTE: This synthetic dataset is designed for educational demonstration
    and algorithm comparison; it is not a substitute for production PCAP traces.
    """
    rng = np.random.default_rng(seed)
    labels = rng.choice([0, 1], size=n, p=[0.78, 0.22])
    rows = []
    for y in labels:
        if y == 0:
            # Normal traffic: standard session duration, moderate packets, few failed logins
            duration = rng.gamma(2.4, 3.5)
            packets = max(1, int(rng.normal(45, 18)))
            byte_count = max(60, int(rng.normal(6500, 2600)))
            failed = int(rng.choice([0, 1], p=[0.95, 0.05]))
            protocol = rng.choice(["TCP", "UDP", "HTTP", "DNS"], p=[0.42, 0.25, 0.25, 0.08])
        else:
            # Suspicious traffic: bursty flows, higher packet/byte volumes, repeated failed logins
            duration = rng.gamma(1.2, 1.4)
            packets = max(1, int(rng.normal(135, 55)))
            byte_count = max(120, int(rng.normal(22000, 8500)))
            failed = int(rng.choice([0, 1, 2, 3, 4, 5], p=[0.10, 0.16, 0.22, 0.22, 0.18, 0.12]))
            protocol = rng.choice(["TCP", "UDP", "HTTP", "DNS"], p=[0.30, 0.20, 0.30, 0.20])
        rows.append([round(float(duration), 3), protocol, packets, byte_count, failed, int(y)])
    return pd.DataFrame(rows, columns=FEATURES + [TARGET])


# Ensure the default synthetic dataset exists
if not os.path.exists(DATA_PATH):
    make_sample_data().to_csv(DATA_PATH, index=False)


def clean_data(df):
    """
    Validate, sanitize, and normalize input network traffic datasets.
    Handles column aliasing, missing features, type coercion, and label verification.
    """
    if df is None or df.empty:
        return None, "The uploaded dataset is empty or could not be read."

    df = df.copy()
    # Normalize column headers
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Map common cybersecurity dataset aliases to standard features
    aliases = {
        "connection_duration": "duration",
        "flow_duration": "duration",
        "dur": "duration",
        "packet_count": "packets",
        "total_packets": "packets",
        "tot_pkts": "packets",
        "bytes_transferred": "bytes",
        "total_bytes": "bytes",
        "tot_bytes": "bytes",
        "failed_login_attempts": "failed_logins",
        "failed_attempts": "failed_logins",
        "failed_login": "failed_logins",
        "proto": "protocol",
        "service": "protocol",
        "attack": "label",
        "class": "label",
        "target": "label",
        "traffic_label": "label",
        "intrusion": "label",
        "is_attack": "label"
    }
    df = df.rename(columns={c: aliases[c] for c in df.columns if c in aliases})

    # Ensure required feature columns exist (apply neutral fallback if missing)
    for col in FEATURES:
        if col not in df.columns:
            if col == "protocol":
                df[col] = "TCP"
            else:
                df[col] = 0

    # Validate target column
    if TARGET not in df.columns:
        return None, (
            "Dataset requires a ground-truth column named 'label' (or alias like 'attack', 'class'). "
            "Values should be 0 (Normal) and 1 (Suspicious)."
        )

    # Normalize labels: map string representations to binary 0/1
    def norm_label(v):
        s = str(v).strip().lower()
        if s in ("1", "1.0", "attack", "anomaly", "malicious", "suspicious", "yes", "true", "intrusion"):
            return 1
        if s in ("0", "0.0", "normal", "benign", "legitimate", "no", "false", "safe"):
            return 0
        try:
            return int(float(s) != 0)
        except Exception:
            return np.nan

    df[TARGET] = df[TARGET].map(norm_label)
    df = df.dropna(subset=[TARGET])

    if len(df) < 10:
        return None, "Dataset must contain at least 10 valid rows for training and evaluation."

    if df[TARGET].nunique() < 2:
        return None, (
            "Dataset must contain examples of both classes (0 = Normal and 1 = Suspicious) "
            "to perform supervised training and evaluation."
        )

    df[TARGET] = df[TARGET].astype(int)

    # Coerce numerical fields
    for c in ["duration", "packets", "bytes", "failed_logins"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Coerce protocol to string
    df["protocol"] = df["protocol"].astype(str).str.strip().str.upper()

    return df, None


def model_pipeline(name, contamination=0.15):
    """
    Construct a scikit-learn Pipeline with proper data preprocessing
    to prevent data leakage (StandardScaler + SimpleImputer for numeric,
    OneHotEncoder for categorical protocols).
    """
    cat_features = ["protocol"]
    num_features = ["duration", "packets", "bytes", "failed_logins"]

    prep = ColumnTransformer([
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler())
            ]),
            num_features
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]),
            cat_features
        )
    ])

    if name == "Decision Tree":
        model = DecisionTreeClassifier(
            max_depth=7,
            class_weight="balanced",
            random_state=42
        )
    elif name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced",
            random_state=42
        )
    else:  # Isolation Forest
        model = IsolationForest(
            contamination=float(contamination),
            random_state=42,
            n_estimators=100
        )

    return Pipeline([("prep", prep), ("model", model)])


def run_model(df, name="Random Forest", contamination=0.15, max_events=120):
    """
    Train and evaluate the selected algorithm with strict train/test split.
    Calculates accuracy, precision, recall, F1, confusion matrix, and generates
    detailed event classifications.
    """
    X = df[FEATURES]
    y = df[TARGET]

    # Stratified test split to prevent data leakage and evaluate on unseen records
    stratify = y if y.value_counts().min() >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )

    pipe = model_pipeline(name, contamination=contamination)

    # Train model on training split
    if name == "Isolation Forest":
        # Unsupervised fitting: Isolation Forest does NOT observe target labels
        pipe.fit(Xtr)
        # Isolation Forest outputs: 1 = Inlier (Normal), -1 = Outlier (Suspicious)
        raw_pred = pipe.predict(Xte)
        pred = (raw_pred == -1).astype(int)
    else:
        # Supervised fitting with training labels
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)

    # Calculate real evaluation metrics on held-out test data
    cm = confusion_matrix(yte, pred, labels=[0, 1]).tolist()
    metrics = {
        "accuracy": round(float(accuracy_score(yte, pred)) * 100, 1),
        "precision": round(float(precision_score(yte, pred, zero_division=0)) * 100, 1),
        "recall": round(float(recall_score(yte, pred, zero_division=0)) * 100, 1),
        "f1": round(float(f1_score(yte, pred, zero_division=0)) * 100, 1),
        "cm": cm,
        "test_rows": int(len(yte))
    }

    # Generate full-dataset classifications for dashboard metrics and event table
    if name == "Isolation Forest":
        pipe_full = model_pipeline(name, contamination=contamination)
        pipe_full.fit(X)
        full_preds = (pipe_full.predict(X) == -1).astype(int)
        raw_scores = -pipe_full.decision_function(X)  # Higher = more anomalous
        min_s, max_s = raw_scores.min(), raw_scores.max()
        if max_s > min_s:
            norm_scores = ((raw_scores - min_s) / (max_s - min_s)) * 100
        else:
            norm_scores = np.full_like(raw_scores, 50.0)
    else:
        pipe_full = model_pipeline(name, contamination=contamination)
        pipe_full.fit(X, y)
        full_preds = pipe_full.predict(X)
        if hasattr(pipe_full.named_steps["model"], "predict_proba"):
            norm_scores = pipe_full.predict_proba(X)[:, 1] * 100
        else:
            norm_scores = full_preds.astype(float) * 100

    # Overall dataset traffic summary
    total_events = len(df)
    suspicious_count = int(np.sum(full_preds == 1))
    normal_count = int(total_events - suspicious_count)
    detection_rate = round(float(suspicious_count / total_events * 100), 1)

    # Event records for interactive table (up to max_events for responsive browser performance)
    event_subset_idx = df.index[:max_events]
    events = []
    for i, idx in enumerate(event_subset_idx):
        row = df.loc[idx]
        is_susp = int(full_preds[idx]) == 1
        ground_truth = "Suspicious" if int(row[TARGET]) == 1 else "Normal"
        events.append({
            "id": f"EV-{str(i + 1).zfill(3)}",
            "duration": round(float(row["duration"]), 2),
            "protocol": str(row["protocol"]),
            "packets": int(row["packets"]),
            "bytes": int(row["bytes"]),
            "failed_logins": int(row["failed_logins"]),
            "prediction": "Suspicious" if is_susp else "Normal",
            "ground_truth": ground_truth,
            "score": round(float(norm_scores[idx]), 1)
        })

    summary = {
        "total_events": total_events,
        "normal_count": normal_count,
        "suspicious_count": suspicious_count,
        "detection_rate": detection_rate,
        "ground_truth_suspicious": int(df[TARGET].sum()),
        "ground_truth_normal": int(len(df) - df[TARGET].sum())
    }

    return metrics, events, summary


@app.route("/")
def index():
    """Render the main security analytics dashboard."""
    try:
        df = pd.read_csv(DATA_PATH)
        row_count = len(df)
    except Exception:
        row_count = 800
    return render_template("index.html", row_count=row_count)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Execute traffic analysis on uploaded CSV or built-in dataset using the chosen algorithm.
    """
    try:
        # Secure file inspection
        if "file" in request.files and request.files["file"].filename:
            uploaded_file = request.files["file"]
            filename = uploaded_file.filename.lower()
            if not filename.endswith(".csv"):
                return jsonify({"error": "Unsupported file format. Please upload a valid CSV (.csv) file."}), 400

            try:
                # Read CSV directly from memory stream (prevents path traversal and disk persistence)
                content = uploaded_file.stream.read().decode("utf-8", errors="replace")
                raw = pd.read_csv(io.StringIO(content))
            except Exception as e:
                return jsonify({"error": f"Failed to parse CSV file: {str(e)}"}), 400
            dataset_name = uploaded_file.filename
        else:
            if not os.path.exists(DATA_PATH):
                make_sample_data().to_csv(DATA_PATH, index=False)
            raw = pd.read_csv(DATA_PATH)
            dataset_name = "Built-in sample dataset"

        df, error = clean_data(raw)
        if error:
            return jsonify({"error": error}), 400

        algo = request.form.get("algorithm", "Random Forest")
        valid_algos = ["Random Forest", "Decision Tree", "Isolation Forest"]
        if algo not in valid_algos:
            algo = "Random Forest"

        try:
            contamination = float(request.form.get("contamination", 0.15))
            contamination = max(0.01, min(0.45, contamination))
        except (ValueError, TypeError):
            contamination = 0.15

        metrics, events, summary = run_model(df, name=algo, contamination=contamination)

        return jsonify({
            "algorithm": algo,
            "dataset_name": dataset_name,
            "summary": summary,
            "metrics": metrics,
            "events": events,
            "contamination": contamination,
            "note": (
                "Evaluation metrics are measured on an isolated, held-out test split (25%). "
                "Suspicious predictions indicate anomalous or risk-correlated patterns identified by the model, "
                "not definitive proof of a hostile network intrusion."
            )
        })

    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@app.route("/api/compare", methods=["POST", "GET"])
def compare():
    """
    Benchmark Decision Tree, Random Forest, and Isolation Forest on identical held-out test splits.
    """
    try:
        if "file" in request.files and request.files["file"].filename:
            uploaded_file = request.files["file"]
            content = uploaded_file.stream.read().decode("utf-8", errors="replace")
            raw = pd.read_csv(io.StringIO(content))
        else:
            if not os.path.exists(DATA_PATH):
                make_sample_data().to_csv(DATA_PATH, index=False)
            raw = pd.read_csv(DATA_PATH)

        df, error = clean_data(raw)
        if error:
            return jsonify({"error": error}), 400

        results = []
        for algo in ["Decision Tree", "Random Forest", "Isolation Forest"]:
            metrics, _, _ = run_model(df, name=algo, contamination=0.15, max_events=1)
            results.append({
                "algorithm": algo,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "cm": metrics["cm"]
            })

        return jsonify({
            "results": results,
            "test_rows": results[0]["accuracy"],  # benchmark metadata
            "note": "Evaluation benchmark performed across all 3 models on the identical held-out 25% test split."
        })

    except Exception as e:
        return jsonify({"error": f"Comparison failed: {str(e)}"}), 500


@app.route("/api/sample")
def sample():
    """Download the built-in synthetic network-traffic CSV dataset."""
    if not os.path.exists(DATA_PATH):
        make_sample_data().to_csv(DATA_PATH, index=False)
    return send_file(
        DATA_PATH,
        as_attachment=True,
        download_name="network_traffic_sample.csv",
        mimetype="text/csv"
    )


@app.route("/api/preview")
def preview():
    """Return the first 10 rows of the current sample dataset for inline inspection."""
    try:
        if not os.path.exists(DATA_PATH):
            make_sample_data().to_csv(DATA_PATH, index=False)
        df = pd.read_csv(DATA_PATH)
        preview_data = df.head(10).to_dict(orient="records")
        return jsonify({
            "columns": list(df.columns),
            "rows": preview_data,
            "total_records": len(df)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "Uploaded file is too large. The maximum permitted size is 8 MB."}), 413


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Requested resource was not found."}), 404


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
