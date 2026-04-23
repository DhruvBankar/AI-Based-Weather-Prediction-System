import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_weather_data(filename='sample_weather.csv', days=365):
    start_date = datetime(2025, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    # Base temperature with seasonal variation (sine wave) + noise
    base_temp = 15 # baseline average
    amplitude = 12 # variation over the year
    
    # Simple sine wave representing a year (lowest in Jan, highest in July)
    # Using day of year for the sine function
    temps = []
    for date in dates:
        day_of_year = date.timetuple().tm_yday
        # shift so peak is around summer (day 200)
        temp_trend = base_temp + amplitude * np.sin(2 * np.pi * (day_of_year - 110) / 365)
        # Add random daily noise
        noise = np.random.normal(0, 3) 
        temps.append(round(temp_trend + noise, 1))

    df = pd.DataFrame({
        'Date': [d.strftime('%Y-%m-%d') for d in dates],
        'Temperature': temps
    })

    # Add a few missing values to test preprocessing
    df.loc[10, 'Temperature'] = np.nan
    df.loc[100, 'Temperature'] = np.nan

    df.to_csv(filename, index=False)
    print(f"Generated {filename} with {days} records.")

if __name__ == "__main__":
    import os
    os.makedirs('dataset', exist_ok=True)
    generate_weather_data('dataset/sample_weather.csv')
