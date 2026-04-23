import os
import io
import base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import hashlib
import json
import joblib
import time
import threading
import uuid
from datetime import datetime, timedelta
from flask import Flask, render_template, request, flash, redirect, url_for, g, jsonify
from flask_cors import CORS
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Setup JSON logging
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, 'request_id'):
            log_record['request_id'] = record.request_id
        if hasattr(record, 'duration'):
            log_record['duration'] = record.duration
        if hasattr(record, 'event'):
            log_record['event'] = record.event
        if hasattr(record, 'status'):
            log_record['status'] = record.status
        return json.dumps(log_record)

logger = logging.getLogger()
logger.setLevel(os.environ.get('LOG_LEVEL', 'INFO').upper())
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
ch = logging.StreamHandler()
ch.setFormatter(JsonFormatter())
logger.addHandler(ch)

# Context filter to inject request_id
class ContextFilter(logging.Filter):
    def filter(self, record):
        record.request_id = getattr(g, 'request_id', 'system')
        return True
logger.addFilter(ContextFilter())

app = Flask(__name__)
CORS(app)

secret_key = os.environ.get('FLASK_SECRET_KEY')
if not secret_key:
    raise ValueError("FLASK_SECRET_KEY environment variable is strictly required for production deployment.")
app.secret_key = secret_key
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_LENGTH', 5 * 1024 * 1024)) # 5 MB upload limit

# Global locks and stores
training_lock = threading.Lock()
metrics_lock = threading.Lock()

system_metrics = {
    'total_requests': 0,
    'cache_hits': 0,
    'model_trainings': 0,
    'total_response_time': 0.0
}

rate_limit_store = {}
RATE_LIMIT = int(os.environ.get('RATE_LIMIT', 30)) # requests per minute per IP

MODEL_VERSION = os.environ.get('MODEL_VERSION', "1.0")
CACHE_DIR = "model_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

@app.before_request
def before_request():
    g.start_time = time.time()
    g.request_id = str(uuid.uuid4())
    
    # Simple Rate Limiting
    ip = request.remote_addr
    now = time.time()
    if ip not in rate_limit_store:
        rate_limit_store[ip] = []
    # Clear old requests
    rate_limit_store[ip] = [t for t in rate_limit_store[ip] if now - t < 60]
    if len(rate_limit_store[ip]) >= RATE_LIMIT:
        logger.warning(f"Rate limit exceeded for IP: {ip}", extra={'event': 'rate_limit_exceeded'})
        return jsonify({"error": "Too Many Requests"}), 429
    rate_limit_store[ip].append(now)

@app.after_request
def after_request(response):
    if request.path in ['/predict', '/', '/health', '/metrics']:
        duration = time.time() - getattr(g, 'start_time', time.time())
        with metrics_lock:
            system_metrics['total_requests'] += 1
            system_metrics['total_response_time'] += duration
        
        logger.info(
            f"{request.method} {request.path}",
            extra={'event': 'request_end', 'duration': round(duration, 4), 'status': response.status_code}
        )
    return response

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/metrics')
def metrics():
    with metrics_lock:
        reqs = system_metrics['total_requests']
        avg_time = system_metrics['total_response_time'] / reqs if reqs > 0 else 0
        return jsonify({
            "total_requests": reqs,
            "cache_hits": system_metrics['cache_hits'],
            "model_trainings": system_metrics['model_trainings'],
            "average_response_time_sec": round(avg_time, 4)
        })

# Configure upload folder
UPLOAD_FOLDER = 'dataset'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))

    if file and file.filename.endswith('.csv'):
        # Compute SHA-256 hash of the uploaded file BEFORE reading into pandas
        try:
            file_bytes = file.read()
            data_hash = hashlib.sha256(file_bytes).hexdigest()
            # Reset file pointer so pandas can read it
            file.seek(0)
            df = pd.read_csv(file)
        except Exception as e:
            logger.error(f"Error reading CSV: {e}", extra={'event': 'csv_error'})
            flash(f"Error reading CSV: {e}")
            return redirect(url_for('index'))

        # Data Validation
        if 'Date' not in df.columns or 'Temperature' not in df.columns:
            flash("CSV must contain 'Date' and 'Temperature' columns.")
            return redirect(url_for('index'))

        if len(df) < 10:
            flash("CSV must contain at least 10 rows for reliable prediction.")
            return redirect(url_for('index'))

        # Preprocessing
        # 1. Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        # Drop rows where Date couldn't be parsed
        df.dropna(subset=['Date'], inplace=True)
        if df.empty or len(df) < 10:
             flash("Not enough valid date entries after parsing.")
             return redirect(url_for('index'))

        # Sort by date
        df.sort_values('Date', inplace=True)
        
        # 2. Handle missing temperature values (Forward fill, then backward fill)
        df['Temperature'] = df['Temperature'].ffill().bfill()
        
        # Prepare data for modeling
        # Convert date to ordinal (number of days) for long-term linear trend
        df['DayIndex'] = (df['Date'] - df['Date'].min()).dt.days
        
        # Advanced Feature Engineering: Trigonometric features for seasonal variations
        df['Sin_Day'] = np.sin(2 * np.pi * df['Date'].dt.dayofyear / 365.25)
        df['Cos_Day'] = np.cos(2 * np.pi * df['Date'].dt.dayofyear / 365.25)
        
        X = df[['DayIndex', 'Sin_Day', 'Cos_Day']]
        y = df['Temperature']
        
        # Multi-Model Caching Logic
        metadata_path = os.path.join(CACHE_DIR, f"{data_hash}_meta.json")
        model_path = os.path.join(CACHE_DIR, f"{data_hash}.joblib")
        
        # Safe handling of concurrent requests
        with training_lock:
            use_cached_model = False
            if os.path.exists(metadata_path) and os.path.exists(model_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    if metadata.get('hash') == data_hash and metadata.get('model_version') == MODEL_VERSION:
                        use_cached_model = True
                except Exception as e:
                    logger.warning(f"Failed to read cache metadata: {e}", extra={'event': 'cache_read_error'})
                    
            if use_cached_model:
                logger.info("Cache hit: Loading model from disk.", extra={'event': 'cache_hit'})
                with metrics_lock:
                    system_metrics['cache_hits'] += 1
                try:
                    load_start = time.time()
                    model = joblib.load(model_path)
                    load_time = time.time() - load_start
                    logger.info(f"Model loaded successfully.", extra={'event': 'model_load', 'duration': round(load_time, 4)})
                    
                    y_pred = model.predict(X)
                    mse = mean_squared_error(y, y_pred)
                    r2 = r2_score(y, y_pred)
                except Exception as e:
                    logger.error(f"Failed to load cached model: {e}", extra={'event': 'model_load_error'})
                    use_cached_model = False # Fallback to retraining
                    
            if not use_cached_model:
                logger.info("Cache miss: Retraining model synchronously.", extra={'event': 'cache_miss'})
                with metrics_lock:
                    system_metrics['model_trainings'] += 1
                train_start = time.time()
                # Train/Test Split for Validation
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train Advanced Model with Cross-Validated Ridge Regularization
                model = make_pipeline(PolynomialFeatures(degree=2), RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]))
                model.fit(X_train, y_train)
                
                # Validation Metrics
                y_test_pred = model.predict(X_test)
                test_mse = mean_squared_error(y_test, y_test_pred)
                test_rmse = np.sqrt(test_mse)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                
                logger.info(f"Model cross-validated. RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}, R2: {test_r2:.2f}")
                
                # Refit on entire dataset for optimal future predictions
                model.fit(X, y)
                train_time = time.time() - train_start
                logger.info(f"Model trained from scratch.", extra={'event': 'model_train', 'duration': round(train_time, 4)})
                
                y_pred = model.predict(X)
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                
                # Persist model and metadata safely
                try:
                    temp_model_path = model_path + '.tmp'
                    joblib.dump(model, temp_model_path)
                    os.replace(temp_model_path, model_path)
                    
                    metadata = {
                        "hash": data_hash,
                        "model_version": MODEL_VERSION,
                        "timestamp": datetime.now().isoformat()
                    }
                    temp_metadata_path = metadata_path + '.tmp'
                    with open(temp_metadata_path, 'w') as f:
                        json.dump(metadata, f)
                    os.replace(temp_metadata_path, metadata_path)
                    logger.info("Model cached successfully.", extra={'event': 'model_cache_save'})
                except Exception as e:
                    logger.error(f"Failed to cache model: {e}", extra={'event': 'model_cache_error'})

        # Post-Request Background Optimization Trigger (Hybrid Approach)
        def background_optimization_task(dataset_hash):
            logger.info(f"Background optimization triggered for {dataset_hash}", extra={'event': 'background_task_start'})
            # Mock long running optimization
            time.sleep(1) 
            logger.info(f"Background optimization completed for {dataset_hash}", extra={'event': 'background_task_complete'})

        threading.Thread(target=background_optimization_task, args=(data_hash,), daemon=True).start()
        
        # Predict next 7 days
        last_date = df['Date'].max()
        last_day_index = df['DayIndex'].max()
        
        future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
        
        # Prepare future features
        future_features = []
        for i, f_date in enumerate(future_dates):
            f_day_index = last_day_index + i + 1
            f_sin = np.sin(2 * np.pi * f_date.dayofyear / 365.25)
            f_cos = np.cos(2 * np.pi * f_date.dayofyear / 365.25)
            future_features.append([f_day_index, f_sin, f_cos])
            
        future_day_df = pd.DataFrame(future_features, columns=['DayIndex', 'Sin_Day', 'Cos_Day'])
        future_predictions = model.predict(future_day_df)
        
        # Visualization
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        
        # Plot historical data
        plt.plot(df['Date'], df['Temperature'], label='Historical Temperature', color='#3498db', linewidth=2)
        
        # Plot trend line (from regression model)
        plt.plot(df['Date'], y_pred, label='Trend Line', color='#e74c3c', linestyle='--', linewidth=2)
        
        # Plot future predictions
        plt.plot(future_dates, future_predictions, label='7-Day Prediction', color='#2ecc71', marker='o', linestyle='-', linewidth=2)
        
        plt.title('Weather Data Analysis & Temperature Prediction', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Temperature', fontsize=12)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot to a base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close()
        
        # Prepare future data for template table
        future_data = []
        for date, temp in zip(future_dates, future_predictions):
            future_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'temp': round(temp, 2)
            })

        return render_template('result.html', 
                               plot_url=plot_url, 
                               mse=round(mse, 2), 
                               r2=round(r2, 2),
                               future_data=future_data)
    else:
        flash('Invalid file format. Please upload a CSV.')
        return redirect(url_for('index'))

if __name__ == '__main__':
    # Production ready deployment using Waitress
    port = int(os.environ.get('PORT', 5000))
    try:
        from waitress import serve
        print(f"Starting production server with Waitress on port {port}...")
        serve(app, host='0.0.0.0', port=port)
    except ImportError:
        print(f"Waitress not installed. Falling back to Flask development server on port {port}...")
        app.run(host='0.0.0.0', debug=os.environ.get('FLASK_DEBUG', 'True') == 'True', port=port)
