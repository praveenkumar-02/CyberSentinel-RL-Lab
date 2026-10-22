"""
CyberSentinel AI — Automated Verification & Integration Test Suite
Validates all backend Flask endpoints, ML model pipelines, upload validation,
and error handling.
"""

import os
import sys
import io
import json

# Ensure CyberSentinel_AI is in path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CYBER_DIR = os.path.join(BASE_DIR, "CyberSentinel_AI")
if os.path.isdir(CYBER_DIR):
    sys.path.insert(0, CYBER_DIR)

from app import app, make_sample_data, clean_data, run_model

def run_tests():
    print("=" * 65)
    print("  CyberSentinel AI — Automated Verification Suite")
    print("=" * 65)
    client = app.test_client()
    passed = 0
    total = 0

    def assert_test(name, condition, details=""):
        nonlocal passed, total
        total += 1
        if condition:
            passed += 1
            print(f"  [PASS] {name}")
        else:
            print(f"  [FAIL] {name} - {details}")

    # 1. Test Homepage
    res = client.get("/")
    assert_test("1. GET / (Homepage loads)", res.status_code == 200 and b"CyberSentinel" in res.data)

    # 2. Test Analyze API - Default (Random Forest)
    res = client.post("/api/analyze", data={"algorithm": "Random Forest", "contamination": "0.15"})
    assert_test("2. POST /api/analyze (Random Forest default)", res.status_code == 200)
    data = res.get_json()
    assert_test("   - Random Forest returns metrics & events", "metrics" in data and "events" in data and len(data["events"]) > 0)
    assert_test("   - Accuracy, Precision, Recall, F1 present", all(k in data["metrics"] for k in ["accuracy", "precision", "recall", "f1", "cm"]))
    assert_test("   - Summary counts verified", data["summary"]["total_events"] > 0 and data["summary"]["normal_count"] + data["summary"]["suspicious_count"] == data["summary"]["total_events"])

    # 3. Test Analyze API - Decision Tree
    res = client.post("/api/analyze", data={"algorithm": "Decision Tree", "contamination": "0.15"})
    assert_test("3. POST /api/analyze (Decision Tree)", res.status_code == 200 and res.get_json()["algorithm"] == "Decision Tree")

    # 4. Test Analyze API - Isolation Forest
    res = client.post("/api/analyze", data={"algorithm": "Isolation Forest", "contamination": "0.22"})
    assert_test("4. POST /api/analyze (Isolation Forest with contamination 0.22)", res.status_code == 200 and res.get_json()["algorithm"] == "Isolation Forest")

    # 5. Test Analyze API - Uploading valid custom CSV
    custom_csv = (
        "duration,protocol,packets,bytes,failed_logins,label\n"
        "10.5,TCP,50,4500,0,0\n"
        "12.1,UDP,60,5500,0,0\n"
        "8.3,HTTP,40,3200,0,0\n"
        "15.2,TCP,70,6100,0,0\n"
        "11.0,DNS,45,4000,0,0\n"
        "9.5,TCP,55,4800,0,0\n"
        "14.1,HTTP,65,5900,0,0\n"
        "1.2,TCP,180,24000,3,1\n"
        "0.8,UDP,200,31000,4,1\n"
        "2.1,HTTP,150,21000,2,1\n"
        "0.5,TCP,220,35000,5,1\n"
        "1.8,DNS,170,26000,3,1\n"
    )
    file_data = {"file": (io.BytesIO(custom_csv.encode("utf-8")), "test_traffic.csv"), "algorithm": "Random Forest"}
    res = client.post("/api/analyze", data=file_data, content_type="multipart/form-data")
    assert_test("5. POST /api/analyze (Upload valid custom CSV)", res.status_code == 200 and res.get_json()["summary"]["total_events"] == 12)

    # 6. Test Upload Validation - Non-CSV extension
    bad_ext_data = {"file": (io.BytesIO(b"malicious script content"), "script.exe")}
    res = client.post("/api/analyze", data=bad_ext_data, content_type="multipart/form-data")
    assert_test("6. POST /api/analyze (Reject non-CSV files)", res.status_code == 400 and "Unsupported file format" in res.get_json().get("error", ""))

    # 7. Test Upload Validation - Missing target label column
    no_label_csv = "duration,protocol,packets,bytes,failed_logins\n10,TCP,50,1000,0\n" * 15
    bad_csv_data = {"file": (io.BytesIO(no_label_csv.encode("utf-8")), "no_label.csv")}
    res = client.post("/api/analyze", data=bad_csv_data, content_type="multipart/form-data")
    assert_test("7. POST /api/analyze (Reject CSV without label column)", res.status_code == 400 and "label" in res.get_json().get("error", ""))

    # 8. Test Upload Validation - Insufficient rows (< 10)
    tiny_csv = "duration,protocol,packets,bytes,failed_logins,label\n10,TCP,50,1000,0,0\n1,TCP,100,5000,2,1\n"
    tiny_data = {"file": (io.BytesIO(tiny_csv.encode("utf-8")), "tiny.csv")}
    res = client.post("/api/analyze", data=tiny_data, content_type="multipart/form-data")
    assert_test("8. POST /api/analyze (Reject insufficient rows < 10)", res.status_code == 400 and "at least 10" in res.get_json().get("error", ""))

    # 9. Test Benchmark API - /api/compare
    res = client.post("/api/compare")
    assert_test("9. POST /api/compare (Benchmark lab)", res.status_code == 200)
    comp_data = res.get_json()
    algos = [r["algorithm"] for r in comp_data.get("results", [])]
    assert_test("   - All 3 algorithms evaluated in benchmark", set(algos) == {"Decision Tree", "Random Forest", "Isolation Forest"})

    # 10. Test Sample Download - /api/sample
    res = client.get("/api/sample")
    assert_test("10. GET /api/sample (Download sample dataset)", res.status_code == 200 and "attachment" in res.headers.get("Content-Disposition", ""))

    # 11. Test Preview API - /api/preview
    res = client.get("/api/preview")
    assert_test("11. GET /api/preview (Inspect first 10 rows)", res.status_code == 200 and len(res.get_json().get("rows", [])) == 10)

    # 12. Test Synthetic Data Generator Function
    df = make_sample_data(n=50, seed=123)
    assert_test("12. make_sample_data (Generates expected columns and rows)", len(df) == 50 and set(df.columns) == {"duration", "protocol", "packets", "bytes", "failed_logins", "label"})

    print("=" * 65)
    print(f"  Test Results: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    print("=" * 65)
    return passed == total

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
