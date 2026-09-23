# 🚀 RackVision AI — Empty Shelf Gap Detector & Operational Warehouse Intelligence

> **Digital Solutions BU — Operational Warehouse Intelligence & Command & Control Center**

RackVision AI is an autonomous edge computer-vision and spatial analytics platform designed for smart retail and warehouse management. It transforms raw warehouse imagery or live surveillance camera streams into real-time shelf inventory intelligence, automated restock queues, and interactive Command & Control (C2) operations.

---

## 🌟 Key Features

* **🤖 Deep Learning Gap Detection:** Powered by Ultralytics **YOLO26m**, trained specifically to detect empty, unstocked shelf spaces (`gap` class).
* **🖥️ Interactive C2 Web Dashboard:** Built with **FastAPI** and modern vanilla JS/CSS frontend, supporting image upload, live webcam streaming, presets, parameter tuning, and real-time visualization.
* **📏 Autonomous Rack & Tier Discovery:** Automatically discovers shelf tiers and rack structures using 1D spatial clustering without needing manual templating or fixed grid configurations.
* **📊 Exact Pixel-Level Analytics:** Calculates accurate Occupancy % vs Vacancy % across shelf tiers using binary pixel mask unions, eliminating duplicate box counting.
* **🚨 Replenishment & AGV Dispatch Engine:** Automatically classifies shelf health urgency (`OPTIMAL`, `MODERATE`, `CRITICAL`) and dispatches replenishment tasks directly to AGV (Automated Guided Vehicle) or worker queues.
* **💻 Command Line Interface (CLI):** Provides standalone script execution for batch processing and automated analytics generation.

---

## 🛠️ Environment Setup & Installation

### 1. Prerequisites
* **Python:** `3.9` or higher recommended
* **Operating System:** Windows / Linux / macOS
* **Virtual Environment:** Python `venv` or `conda`

### 2. Clone & Setup Virtual Environment

On Windows (PowerShell):
```powershell
# Navigate into the project directory
cd .\empty-shelves\

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\activate
```

On Linux / macOS:
```bash
cd empty-shelves
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Model Weights
Ensure the pre-trained model weights file `empty-shelves-yolo26m.pt` is placed in the `models/` directory:
```
empty-shelves/
└── models/
    ├── empty-shelves-yolo26m.pt  (Primary YOLO26 Model)
    └── empty-shelves-yolo26s.pt  (Small Variant)
```

---

## 🚀 How to Execute & Run

### A. Launching the Web Server & C2 Dashboard

Start the FastAPI application using **Uvicorn**:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Once started, open your web browser and navigate to:
👉 **`http://127.0.0.1:8000`**
