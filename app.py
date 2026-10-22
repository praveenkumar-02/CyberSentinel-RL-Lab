"""
CyberSentinel AI — Root Application Launcher
Allows launching CyberSentinel AI directly from the workspace root.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NESTED_DIR = os.path.join(BASE_DIR, "CyberSentinel_AI")

if os.path.isdir(NESTED_DIR):
    sys.path.insert(0, NESTED_DIR)

from app import app

if __name__ == "__main__":
    print("=" * 65)
    print("  CyberSentinel AI — Intelligent Cyber Threat Detection Platform")
    print("  Starting server at: http://127.0.0.1:5000")
    print("=" * 65)
    app.run(host="127.0.0.1", port=5000, debug=True)
