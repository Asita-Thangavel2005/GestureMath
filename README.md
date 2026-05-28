# GestureMath

An AI-powered real-time air-writing mathematical expression solver using hand gesture recognition, computer vision, and multimodal AI.

Users can draw equations in the air using their fingers, and the system recognizes and solves the mathematical expression automatically.

---

# Features

* ✋ Real-time hand tracking using MediaPipe
* 🎨 Air drawing with gesture-based interaction
* 🧠 AI-powered handwritten math recognition using Llama Vision
* ➗ Mathematical evaluation using SymPy
* 🗑 Gesture-controlled stroke erase
* ⚡ Smooth interpolated drawing system
* 📷 Live webcam processing with OpenCV
* 🧹 Canvas clearing and dynamic stroke management

---

# Tech Stack

## Computer Vision

* OpenCV
* MediaPipe

## AI / Recognition

* Groq API
* Llama 4 Vision Model

## Mathematics Engine

* SymPy

## Core Language

* Python

---

# System Workflow

1. Webcam captures live video feed
2. MediaPipe detects hand landmarks
3. Gestures control drawing actions
4. Finger movement creates air-written equations
5. Canvas image is preprocessed
6. Image sent to Llama Vision model for recognition
7. Extracted equation parsed using SymPy
8. Final mathematical result displayed in real time

---

# Gestures

| Gesture            | Action              |
| ------------------ | ------------------- |
| ☝️ Index Finger Up | Draw                |
| ✊ Fist             | Pause / Save Stroke |
| ✌️ Two Fingers Up  | Erase Last Stroke   |
| 👍 Thumbs Up       | Reserved            |
| `s` Key            | Solve Equation      |
| `c` Key            | Clear Canvas        |
| `q` Key            | Quit Application    |

---

# Project Structure

```bash
GestureMath/
│
├── main.py
├── config.py
│
├── modules/
│   ├── hand_tracker.py
│   ├── canvas_manager.py
│   ├── gesture_recognizer.py
│   └── solver.py
│
├── requirements.txt
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/Asita-Thangavel2005/GestureMath.git
cd GestureMath
```

## Create Virtual Environment (Optional)

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / Mac

```bash
source venv/bin/activate
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Requirements

Main libraries used:

```txt
opencv-python
mediapipe
numpy
sympy
pillow
groq
```

---

# API Setup

Add your Groq API key inside `solver.py`

```python
GROQ_API_KEY = "your_api_key_here"
```

Get API key from:
https://console.groq.com/

---

# Run the Project

```bash
python main.py
```

---
