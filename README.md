# 🌤️ Atmospheric: Advanced Weather Prediction & Simulation Engine

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Production-lightgrey.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
**Atmospheric** is a production-grade, full-stack data science application that beautifully bridges the gap between complex machine learning and immersive front-end design. 

The system allows users to upload historical weather data (CSV) and utilizes a **Cross-Validated Ridge Regression** model to forecast temperatures for the upcoming week. Rather than just returning numbers, the application instantly transports the user into a dynamic, 60-FPS hardware-accelerated Canvas environment that reacts to their scroll position, transitioning seamlessly from a sunny morning to a rainy midnight.

It is unique because it demonstrates how to encapsulate heavy Python machine learning pipelines inside a hyper-optimized, visually stunning, and highly secure web architecture.

---

## ✨ Key Features

### 🎨 Frontend: Immersive & Reactive
* **Real-Time Animated Canvas**: Pure Canvas2D engine generating dynamic weather particles (rain, clouds, stars) without heavy CSS/DOM hacks.
* **Scroll-Driven Transitions**: The environment responds directly to the user's scroll depth, smoothly interpolating atmospheric colors and densities via a custom cubic-easing state controller.
* **Hardware-Accelerated Parallax**: Fluid mouse-tracking depth effects that decouple cleanly from the main execution thread.
* **60 FPS Guarantee**: Aggressively optimized object pooling and `requestAnimationFrame` debouncing.

### ⚙️ Backend: Robust & Scalable
* **ML-Based Temperature Prediction**: Leverages `scikit-learn` Polynomial Features and Trigonometric Seasonality (Sin/Cos) to map complex, non-linear historical data.
* **SHA-256 Dataset Caching**: Hashes incoming files instantly. Identical datasets bypass the ML pipeline entirely, returning serialized `joblib` models in milliseconds.
* **Hybrid Processing Model**: Executes primary training synchronously for instant UI feedback, while offloading background optimizations to daemon threads.
* **Production-Ready Security**: In-memory rate limiting, 5MB rigid upload constraints, and strict payload validation protect the server from abuse.

---

## 🛠️ Tech Stack
* **Backend:** Python, Flask, Waitress (WSGI), Joblib, Hashlib
* **Machine Learning:** Scikit-Learn, Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Frontend:** HTML5, CSS3, Vanilla JavaScript (ES6+ Canvas API)

---

## 🏗️ System Architecture
1. **Data Ingestion**: A user uploads a CSV file. The backend reads the byte-stream and generates a SHA-256 hash *before* invoking Pandas, ensuring memory is only consumed if absolutely necessary.
2. **Caching Layer**: Using a "Double-Checked Lock" concurrency pattern, the server checks the `model_cache` directory. If the hash exists, it instantly deserializes the model. If it's a miss, it acquires a thread lock to safely execute the heavy ML pipeline.
3. **Prediction Engine**: The ML pipeline engineers time-series features (Ordinal Days + Trigonometric Seasonality) and fits a `RidgeCV` model with automatic hyperparameter tuning to forecast the next 7 days.
4. **Presentation**: The backend returns a dynamic `result.html` payload. The frontend JavaScript detects the response, unlocks the UI, and initializes the high-performance Canvas weather engine for the user to explore the data.

---

## ⚡ Performance Highlights
* **Lock-Free Cache Reads**: Background threads waiting on the main ML `training_lock` will instantly abort and read from the disk cache if another thread finishes the identical task milliseconds prior.
* **Event Debouncing**: Notoriously expensive DOM events (`scroll` and `mousemove`) are throttled inside `requestAnimationFrame` wrappers, guaranteeing layout calculations never exceed the monitor's native refresh rate.
* **Tab Inactivity Sleeping**: Injects `document.hidden` into the core render loop. Switching tabs completely suspends the Canvas engine, dropping CPU usage to 0%.
* **Memory-Safe Particle Pooling**: Raindrops and cloud layers recycle memory addresses rather than instantiating new objects, completely bypassing V8 Garbage Collection lag spikes.

---

## 🚀 How to Run Locally

### 1. Requirements
Ensure you have Python 3.10+ installed.

### 2. Setup
Clone the repository and set up a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure & Run
You must provide a secure session key to start the Flask application:
```bash
# Windows PowerShell
$env:FLASK_SECRET_KEY="local_dev_key"
python app.py

# Mac/Linux
export FLASK_SECRET_KEY="local_dev_key"
python app.py
```
Open `http://localhost:5000` in your browser. Upload the `dataset/sample_weather.csv` to test!

---

## 🌍 Deployment

This application uses **Waitress** as its production WSGI server, making it inherently ready for platforms like Render or Railway.

### Render / Railway (PaaS)
1. Push your repository to GitHub.
2. Create a new "Web Service" on your PaaS dashboard and connect the repo.
3. The platform will automatically detect `requirements.txt` and install dependencies.
4. The included `Procfile` executes the start command:
   ```bash
   waitress-serve --port=$PORT --host=0.0.0.0 app:app
   ```
5. **Crucial**: Add `FLASK_SECRET_KEY` to your environment variables on the dashboard to allow the server to boot securely.

---

## 📸 Demo

> *(Replace the links below with your actual assets when you record a screen capture)*

### Interactive Weather Engine
![Weather UI Demo](docs/assets/weather_demo.gif)
*The environment seamlessly transitions based on scroll depth.*

### Prediction Dashboard
![Dashboard Screenshot](docs/assets/dashboard.png)
*Beautiful Matplotlib projections securely delivered to the frontend.*
