# Face Reconstruction Backend (Python 3.10)

This repository contains the backend service for a Face Recognition application built with **FastAPI**, **dlib**, **OpenCV**, and **PyTorch**. The project is **strictly configured for Python 3.10** to ensure compatibility with all required packages, especially `dlib`.

---

## ✅ Requirements

* **Python 3.10.x** (required)
* Windows (for the precompiled `dlib` wheel)
* Git

> ⚠️ Do **not** use Python 3.11 or later — `dlib` and some dependencies will fail to install.

---

## 📦 Python Dependencies

All dependencies are pinned in `requirements.txt` for reproducible installs.

A custom precompiled wheel for dlib (Windows + Python 3.10) is used:

```
https://huggingface.co/hanamizuki-ai/pypi-wheels/resolve/a056889004afa48f7b178e35846788eb72002073/dlib/dlib-19.24.1-cp310-cp310-win_amd64.whl
```

---

## 🛠️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

---

### 2️⃣ Create a Python 3.10 Virtual Environment

Make sure Python 3.10 is installed and available in PATH.

```bash
python -m venv .venv310
```

---

### 3️⃣ Activate the Virtual Environment

**PowerShell (Windows):**

```powershell
.\.venv310\Scripts\Activate.ps1
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

---

### 4️⃣ Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

✅ This will install all packages including **dlib** using the precompiled wheel.

---

## 🚀 Run the Development Server

Navigate to the backend directory (if applicable):

```bash
cd backend
```

Start the FastAPI server using Uvicorn:

```bash
uvicorn app:app --reload
```

* Server runs at: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🧠 Tech Stack

* **FastAPI** – Backend framework
* **Uvicorn** – ASGI server
* **dlib + face-recognition** – Face detection & recognition
* **OpenCV** – Image processing
* **PyTorch** – Deep learning models
* **Ultralytics (YOLO)** – Object detection
* **NumPy / SciPy / Matplotlib** – Numerical & visual processing

---

## 🧩 Common Issues

### ❌ dlib installation fails

* Ensure **Python 3.10** is installed
* Do not remove the custom dlib wheel from `requirements.txt`
* Only works on **Windows (64-bit)**

### ❌ `uvicorn` command not found

```bash
pip install uvicorn
```

(make sure the virtual environment is activated)

---

## ✅ Notes

* Always activate `.venv310` before running the server
* Dependencies are tightly version-locked — avoid upgrading blindly
* For production, disable `--reload`
