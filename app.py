import os
import requests
from flask import Flask, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

ALPHA_VANTAGE_API_KEY = "7QN13O36KPJM2DLV"
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"


# Fetch stock data function
def fetch_stock_data(symbol):
    url = f"{ALPHA_VANTAGE_URL}?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url)
    data = response.json()
    if 'Time Series (Daily)' not in data:
        return None

    df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
    df = df.astype(float)
    df['date'] = pd.to_datetime(df.index)
    df = df[['date', '4. close']]  # Close price
    df = df.rename(columns={'4. close': 'price'})
    df = df.sort_values('date')  # Ensure the data is sorted by date
    return df


# Prediction function for next year
def predict_stock_prices(df):
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['price'].values

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Predict for the next year (365 days)
    future_days = 90
    future_X = np.arange(len(df), len(df) + future_days).reshape(-1, 1)
    predictions = model.predict(future_X)

    # Generate future dates
    last_date = df['date'].iloc[-1]
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, future_days + 1)]
    return future_dates, predictions


@app.route('/')
def index():
    """Serve the frontend."""
    return render_template('index.html')

# Fetch exchange rate
def get_usd_to_inr_rate():
    url = "https://api.exchangerate-api.com/v4/latest/USD"
    response = requests.get(url)
    data = response.json()
    return data['rates']['INR']

# Fetch stock data function
def fetch_stock_data(symbol):
    usd_to_inr_rate = get_usd_to_inr_rate()  # Get USD to INR conversion rate
    url = f"{ALPHA_VANTAGE_URL}?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url)
    data = response.json()
    if 'Time Series (Daily)' not in data:
        return None

    df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
    df = df.astype(float)
    df['date'] = pd.to_datetime(df.index)
    df = df[['date', '4. close']]  # Close price
    df = df.rename(columns={'4. close': 'price'})
    df['price'] = df['price'] * usd_to_inr_rate  # Convert to INR
    df = df.sort_values('date')  # Ensure the data is sorted by date
    return df

@app.route('/stockData/<symbol>')
def stock_data(symbol):
    df = fetch_stock_data(symbol)
    if df is None:
        return jsonify({'error': 'Data not available for this stock.'}), 404

    # Filter the last 1 year of data for the graph
    one_year_data = df[df['date'] >= (df['date'].max() - pd.Timedelta(days=365))]

    # Predict stock prices for the next year
    future_dates, predictions = predict_stock_prices(df)

    return jsonify({
        'name': symbol,
        'price': df.iloc[-1]['price'],
        'change': df.iloc[-1]['price'] - df.iloc[-2]['price'],
        'dates': one_year_data['date'].dt.strftime('%Y-%m-%d').tolist(),
        'prices': one_year_data['price'].tolist(),
        'prediction_dates': [date.strftime('%Y-%m-%d') for date in future_dates],
        'predicted_prices': predictions.tolist(),
    })


if __name__ == '__main__':
    app.run(debug=True)
