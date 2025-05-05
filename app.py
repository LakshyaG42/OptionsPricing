from flask import Flask, render_template, request, jsonify
app = Flask(__name__)

import json
import plotly
import plotly.graph_objs as go
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
# Models
from models.black_scholes import price_option
from models.binomial import binomial_price
from models.pde import crank_nicolson_call, crank_nicolson_put


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/plot", methods=["POST"])
def route_plot():
    data = request.json
    model = data.get("model", "black_scholes")

    if model == "black_scholes":
        return jsonify(plot_black_scholes(data))
    
    elif model == "binomial":
        return jsonify(plot_binomial(data))
        #return jsonify({"error": "Binomial plot not implemented yet."})
    
    elif model == "monte_carlo":
        # return jsonify(plot_monte_carlo(data))
        return jsonify({"error": "Monte Carlo plot not implemented yet."})

    elif model == "pde":
        return jsonify(plot_pde(data))
        #return jsonify({"error": "PDE plot not implemented yet."})

    else:
        return jsonify({"error": f"Unknown model: {model}"}), 400

@app.route("/historical_price", methods=["POST"])
def route_historical_price():
    data = request.json
    ticker_str = data.get("ticker")
    quote_date_str = data.get("quote_date")
    expiry_date_str = data.get("expiry_date")
    K = data.get("K")
    option_type = data.get("option_type", "call")
    model = data.get("model", "black_scholes")

    # Basic Input Validation
    if not all([ticker_str, quote_date_str, expiry_date_str, K]):
        return jsonify({"error": "Missing required historical parameters."}), 400

    try:
        quote_date = datetime.strptime(quote_date_str, '%Y-%m-%d')
        expiry_date = datetime.strptime(expiry_date_str, '%Y-%m-%d')

        if expiry_date <= quote_date:
             return jsonify({"error": "Expiry date must be after quote date."}), 400

        # --- Fetch Data & Calculate Parameters ---
        ticker = yf.Ticker(ticker_str)

        # 1. Get Stock Price (S) on quote_date
        hist_s = ticker.history(start=quote_date, end=quote_date + timedelta(days=1), auto_adjust=False)
        if hist_s.empty or 'Close' not in hist_s.columns:
             hist_s = ticker.history(end=quote_date + timedelta(days=1), period="5d", auto_adjust=False)
             if hist_s.empty:
                 return jsonify({"error": f"Could not fetch stock price for {ticker_str} around {quote_date_str}."}), 400
             S = hist_s['Close'].iloc[-1]
        else:
             S = hist_s['Close'].iloc[0]

        # 2. Calculate Volatility (sigma)
        start_vol_date = quote_date - timedelta(days=365)
        hist_vol = ticker.history(start=start_vol_date - timedelta(days=5), end=quote_date, auto_adjust=False)
        if len(hist_vol) < 2:
             return jsonify({"error": f"Not enough historical data for {ticker_str} to calculate volatility before {quote_date_str}."}), 400

        hist_vol['LogReturn'] = np.log(hist_vol['Close'] / hist_vol['Close'].shift(1))
        daily_std_dev = hist_vol['LogReturn'].std()
        sigma = daily_std_dev * np.sqrt(252)

        if np.isnan(sigma) or sigma == 0:
             return jsonify({"error": f"Could not calculate valid volatility for {ticker_str}."}), 400

        # 3. Calculate Time to Maturity (T)
        T = (expiry_date - quote_date).days / 365.0
        if T <= 0: # Add check for non-positive time to maturity
            return jsonify({"error": "Time to maturity must be positive."}), 400

        # 4. Fetch Risk-Free Rate (r) using 13-Week Treasury Bill (^IRX)
        r = 0.05 # Default value
        try:
            irx = yf.Ticker("^IRX")
            # Fetch data around the quote date, look back 5 days
            hist_r = irx.history(end=quote_date + timedelta(days=1), period="5d", auto_adjust=False)
            if not hist_r.empty and 'Close' in hist_r.columns:
                # Use the last available closing rate on or before the quote date
                r_percentage = hist_r['Close'].iloc[-1]
                r = r_percentage / 100.0 # Convert percentage to decimal
            else:
                 app.logger.warning(f"Could not fetch risk-free rate (^IRX) around {quote_date_str}. Using default 0.")
                 # Optionally, return an error or use a fallback default
                 # return jsonify({"error": f"Could not fetch risk-free rate (^IRX) around {quote_date_str}."}), 400

        except Exception as r_err:
            app.logger.error(f"Error fetching risk-free rate (^IRX): {r_err}")
            # Optionally, return an error or use a fallback default
            # return jsonify({"error": "Failed to fetch risk-free rate."}), 500
            r = 0.05 # Fallback to fixed rate on error
            app.logger.warning(f"Using fallback risk-free rate: {r}")

        # --- Prepare data for plotting function ---
        plot_data = {
            "S": S,
            "K": K,
            "T": T,
            "r": r,
            "sigma": sigma,
            "option_type": option_type,
        }

        # --- Call Pricing Model & Generate Plot ---
        price = None
        plot_json = None
        plot_result = {}

        if model == "black_scholes":
            plot_result = plot_black_scholes(plot_data)
            price = plot_result.get("price")
            plot_json = plot_result.get("plot")
        elif model == "binomial":
            n_steps = data.get("n_steps", 100)
            american = False
            plot_data["n_steps"] = n_steps
            plot_data["exercise_style"] = 'european'
            plot_result = plot_binomial(plot_data)
            price = plot_result.get("price")
            plot_json = plot_result.get("plot")
        else:
             return jsonify({"error": f"Model '{model}' not supported for historical pricing/plotting yet."}), 400

        return jsonify({
            "S": S,
            "sigma": sigma,
            "T": T,
            "r": r,
            "price": price,
            "plot": plot_json
        })

    except ValueError as ve:
         app.logger.error(f"Date parsing error: {ve}")
         return jsonify({"error": "Invalid date format. Please use YYYY-MM-DD."}), 400
    except Exception as e:
        app.logger.error(f"Error in /historical_price: {e}")
        return jsonify({"error": f"An error occurred processing the historical data: {str(e)}"}), 500

def plot_black_scholes(data):
    K = float(data["K"])
    T = data["T"]
    r = data["r"]
    sigma = data["sigma"]
    option_type = data["option_type"]
    S_user = data["S"]

    s_min = max(5, int(S_user * 0.5))
    s_max = int(S_user * 1.5)
    step = max(1, int((s_max - s_min) / 20))
    S_range = list(range(s_min, s_max + step, step))

    prices = [price_option(S_val, K, T, r, sigma, option_type) for S_val in S_range]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S_range, y=prices, mode='lines+markers', name='Option Price'))
    fig.add_trace(go.Scatter(x=[S_user], y=[price_option(S_user, K, T, r, sigma, option_type)],
                             mode='markers', marker=dict(color='red', size=10), name='Current S'))
    fig.update_layout(title='Black-Scholes Option Price',
                      xaxis_title='Stock Price (S)',
                      yaxis_title='Option Price')

    price_at_S = price_option(S_user, K, T, r, sigma, option_type)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return {"plot": graphJSON, "price": price_at_S}


def plot_binomial(data):
    S_user = data["S"]
    K = float(data["K"])
    T = data["T"]
    r = data["r"]
    sigma = data["sigma"]
    option_type = data.get("option_type", "call")
    n_steps = data.get("n_steps", 100)
    american = data.get("exercise_style", 'european') == 'american'

    s_min = max(5, int(S_user * 0.5))
    s_max = int(S_user * 1.5)
    step = max(1, int((s_max - s_min) / 20))
    S_range = list(range(s_min, s_max + step, step))

    prices = [
        binomial_price(S_val, K, T, r, sigma, n_steps=n_steps, option_type=option_type, american=american)
        for S_val in S_range
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S_range, y=prices, mode='lines+markers', name='Binomial Price'))
    user_price = binomial_price(S_user, K, T, r, sigma, n_steps=n_steps, option_type=option_type, american=american)
    fig.add_trace(go.Scatter(x=[S_user], y=[user_price],
                             mode='markers', marker=dict(color='red', size=10), name='Current S'))
    fig.update_layout(title='Binomial Option Price',
                      xaxis_title='Stock Price (S)',
                      yaxis_title='Option Price')

    return {
        "plot": json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder),
        "price": user_price
    }

def plot_pde(data):
    S_user = data["S"]
    K = data["K"]
    T = data["T"]
    r = data["r"]
    sigma = data["sigma"]
    x_max = data.get("x_max", 3.0)
    n_t = data.get("n_t", 1000)

    option_type = data.get("option_type", "call")
    if option_type == "call":
        x, V, user_price = crank_nicolson_call(S_user, K, sigma, T, r, x_max=x_max, N_t=n_t)
    else:
        x, V, user_price = crank_nicolson_put(S_user, K, sigma, T, x_max=x_max, N_t=n_t)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=V, mode='lines+markers', name='CN Price'))
    fig.update_layout(title='Crank–Nicolson Option Price',
                      xaxis_title='Stock Price (S)',
                      yaxis_title='Option Price')

    return {
        "plot": json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder),
        "price": user_price
    }


if __name__ == "__main__":
    app.run(debug=True)
