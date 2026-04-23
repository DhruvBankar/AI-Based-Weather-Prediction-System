# Weather Data Analysis & Temperature Prediction System

A production-ready Data Science and Web Application project built with Flask and Scikit-Learn.

## Features
- **Upload CSV Dataset**: Users can upload historical daily weather data seamlessly.
- **Data Preprocessing**: Automatically handles missing values and date conversions robustly.
- **Data Visualization**: Generates a dynamic graph plotting historical trends alongside a predicted trendline.
- **Predictive Modeling**: Uses a Polynomial RidgeCV Pipeline with Trigonometric Seasonal Features to forecast the temperature for the next 7 days.
- **Metrics Display**: Displays MSE and R² natively, with MAE and RMSE logged via JSON.
- **Multi-Model Caching**: Uses SHA-256 hashing to instantly cache and retrieve models without redundant training.
- **Production Safety**: Includes Rate Limiting, File Size Limits, Request Logging, and Thread Locks.
- **Clean Dashboard UI**: Modern, responsive design utilizing Hardware-accelerated Canvas.

## Tech Stack
- **Backend**: Python, Flask, Waitress, Joblib
- **Data Science / ML**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Frontend**: HTML5, CSS3, Vanilla JavaScript

## Project Structure
```text
.
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── Procfile                    # Deployment execution command
├── README.md                   # Project documentation
├── dataset/
│   └── sample_weather.csv      # Synthetic sample dataset for testing
├── templates/
│   ├── index.html              # Upload page UI
│   └── result.html             # Results dashboard UI
└── static/
    ├── style.css               # Core styling
    ├── weather.css             # UI Weather Theme
    └── js/weatherSystem.js     # Canvas Engine
```

## Environment Variables

For production, the following variables can be configured:
- `PORT`: Port to run the server on (default: `5000`)
- `FLASK_SECRET_KEY`: Secret key for session/flash (default: `super_secret_key_for_flask`)
- `LOG_LEVEL`: Application logging level (default: `INFO`)
- `MAX_CONTENT_LENGTH`: Maximum allowed upload size in bytes (default: `5242880` - 5MB)
- `RATE_LIMIT`: Max requests per minute per IP (default: `30`)
- `MODEL_VERSION`: Tag attached to cached models (default: `1.0`)

## How to Run Locally

1. **Clone or Download the Repository**
2. **Navigate to the Project Directory**
3. **Create a Virtual Environment (Optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
5. **Run the Application**
   ```bash
   python app.py
   ```
6. **Open in Browser**
   Navigate to `http://localhost:5000`

## Production Deployment Targets

The application natively uses `Waitress`, a production-grade WSGI server perfectly suited for Windows and Linux environments.

### Option A: Render
1. Push your code to a GitHub repository.
2. Log in to [Render](https://render.com/) and create a new **Web Service**.
3. Connect your repository.
4. Set the **Build Command** to: `pip install -r requirements.txt`
5. Set the **Start Command** to: `waitress-serve --port=$PORT --host=0.0.0.0 app:app` (or rely on the `Procfile` already provided).
6. Under **Environment Variables**, Render automatically injects `PORT`. Add any custom variables like `RATE_LIMIT` or `LOG_LEVEL`.

### Option B: Railway
1. Push your code to GitHub.
2. Go to [Railway](https://railway.app/) and create a new project from your GitHub repo.
3. Railway will automatically detect the Python environment and run `pip install -r requirements.txt`.
4. Railway will automatically read the provided `Procfile` and use it as the start command.
5. In the **Variables** tab, you can easily configure `FLASK_SECRET_KEY` and other overrides.

### Option C: AWS EC2 (Ubuntu/Amazon Linux)
1. SSH into your EC2 instance and install Python 3.10+:
   ```bash
   sudo apt update && sudo apt install python3-pip python3-venv
   ```
2. Clone the repository and navigate into it.
3. Set up the virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
4. Export environment variables:
   ```bash
   export PORT=80
   export FLASK_SECRET_KEY="your-secure-random-key"
   ```
5. Run the server in production mode using `Waitress` (requires `sudo` for port 80):
   ```bash
   sudo -E venv/bin/python app.py
   ```
   *(For continuous execution, use a systemd service or `pm2`)*

## Health & Monitoring
- **GET `/health`**: Returns `{"status": "healthy"}` for deployment load balancer probing.
- **GET `/metrics`**: Returns lightweight statistics including `total_requests`, `cache_hits`, `model_trainings`, and `average_response_time_sec`.
