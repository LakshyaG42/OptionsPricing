from flask import Flask, render_template, request, jsonify
from flask_caching import Cache
app = Flask(__name__)

# For this project SimpleCache is fine, but for production I would use FileSystemCache or RedisCache.
app.config['CACHE_TYPE'] = 'SimpleCache'
cache = Cache(app)

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
#from models.gbm import simulate_gbm
from models.monte_carlo import MonteCarloOptionPricer

@cache.memoize(timeout=3600)  # Cache for 1 hour
def get_stock_history(ticker_symbol, start_date_str, end_date_str):
    app.logger.info(f"Fetching stock history for {ticker_symbol} from {start_date_str} to {end_date_str}")
    ticker = yf.Ticker(ticker_symbol)
    return ticker.history(start=start_date_str, end=end_date_str, auto_adjust=False)


@cache.memoize(timeout=3600)  # Cache for 1 hour
def get_stock_history_period(ticker_symbol, end_date_str, period_str):
    app.logger.info(f"Fetching stock history for {ticker_symbol}, end date {end_date_str}, period {period_str}")
    ticker = yf.Ticker(ticker_symbol)
    return ticker.history(end=end_date_str, period=period_str, auto_adjust=False)


@cache.memoize(timeout=86400)  # Cache for 1 day
def get_risk_free_rate_history(end_date_str, period_str):
    app.logger.info(f"Fetching risk-free rate history, end date {end_date_str}, period {period_str}")
    irx = yf.Ticker("^IRX")
    return irx.history(end=end_date_str, period=period_str, auto_adjust=False)


@app.route("/")
def home():
    return render_template("index.html")


def plot_pnl_histogram(pnl_counts, pnl_bin_edges, model_name):
    """Generates a Plotly JSON for the P&L histogram."""
    fig = go.Figure(data=[go.Bar(
        x=[(pnl_bin_edges[i] + pnl_bin_edges[i+1])/2 for i in range(len(pnl_counts))],  # Bin centers
        y=pnl_counts,
        marker_color='#EF553B'
    )])
    fig.update_layout(
        title=f"P&L Distribution (vs. {model_name} Price)",
        xaxis_title="Profit/Loss ($)",
        yaxis_title="Frequency (Number of Simulations)",
        bargap=0.1,
        template='plotly_white'
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def generate_full_analytics_package(data, model_price, model_name_str, num_paths_sim=1000, num_steps_sim=100):
    """
    Helper to generate GBM simulation, its plot, MC price, P&L histogram, and stats.
    model_price is the price from the primary model (BS, Binomial, PDE).
    """
    S_user = data["S"]
    K = float(data["K"])
    T = data["T"]
    r = data["r"]
    sigma = data["sigma"]
    option_type = data["option_type"]

    # 1. GBM Path Simulation Plot
    gbm_simulation_plot_json = plot_gbm_simulation(S_user, K, T, r, sigma, steps=num_steps_sim, num_paths=100)  # Keep this to 100 paths for viz

    # 2. Run more simulations for analytics
    time_array_analytics, price_paths_analytics = simulate_gbm_paths(S_user, T, r, sigma, steps=num_steps_sim, num_paths=num_paths_sim)
    terminal_prices_analytics = price_paths_analytics[-1, :]
    payoffs_analytics = calculate_payoffs(terminal_prices_analytics, K, option_type)
    
    # 3. Monte Carlo Price from these simulations
    mc_price_sim = np.mean(np.exp(-r * T) * payoffs_analytics)

    # 4. P&L analysis and Summary Stats
    # Use the primary model's price as the option_cost for P&L
    summary_stats, pnl_counts, pnl_bin_edges = get_gbm_analytics(
        terminal_prices_analytics, 
        payoffs_analytics, 
        K, T, r, option_type, 
        option_cost=model_price 
    )
    
    # 5. P&L Histogram Plot
    pnl_histogram_plot_json = plot_pnl_histogram(pnl_counts, pnl_bin_edges, model_name_str)

    return {
        "gbm_simulation_plot": gbm_simulation_plot_json,
        "monte_carlo_price_from_analytics_sim": mc_price_sim,
        "gbm_summary_stats": summary_stats,
        "pnl_histogram_plot": pnl_histogram_plot_json
    }


@app.route("/plot", methods=["POST"])
def route_plot():
    data = request.json
    model = data.get("model", "black_scholes")
    if "K" in data:
        data["K"] = float(data["K"])
    
    # Default values for analytics simulations
    num_paths_analytics = data.get("n_simulations", 1000)  # Use n_simulations if provided, else 1000
    num_steps_analytics = data.get("n_steps", 100)  # Use n_steps if provided, else 100

    if model == "black_scholes":
        result = plot_black_scholes(data)
        analytics_package = generate_full_analytics_package(data, result["price"], "Black-Scholes", num_paths_sim=num_paths_analytics, num_steps_sim=num_steps_analytics)
        result.update(analytics_package)
        return jsonify(result)

    elif model == "binomial":
        result = plot_binomial(data)
        analytics_package = generate_full_analytics_package(data, result["price"], "Binomial", num_paths_sim=num_paths_analytics, num_steps_sim=num_steps_analytics)
        result.update(analytics_package)
        return jsonify(result)

    elif model == "monte_carlo":
        return jsonify(plot_monte_carlo(data))

    elif model == "pde":
        result = plot_pde(data)
        analytics_package = generate_full_analytics_package(data, result["price"], "PDE", num_paths_sim=num_paths_analytics, num_steps_sim=num_steps_analytics)
        result.update(analytics_package)
        return jsonify(result)

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
    exercise_style = data.get("exercise_style", "european")
    
    # Default values for analytics simulations from historical context
    num_paths_analytics = data.get("n_simulations", 1000) 
    num_steps_analytics = data.get("n_steps", 100)

    if not all([ticker_str, quote_date_str, expiry_date_str, K]):
        return jsonify({"error": "Missing required historical parameters."}), 400

    try:
        K = float(K)
        quote_date = datetime.strptime(quote_date_str, '%Y-%m-%d')
        expiry_date = datetime.strptime(expiry_date_str, '%Y-%m-%d')

        if expiry_date <= quote_date:
            return jsonify({"error": "Expiry date must be after quote date."}), 400

        hist_s = get_stock_history(ticker_str, quote_date.strftime('%Y-%m-%d'), (quote_date + timedelta(days=1)).strftime('%Y-%m-%d'))
        if hist_s.empty or 'Close' not in hist_s.columns:
            hist_s = get_stock_history_period(ticker_str, (quote_date + timedelta(days=1)).strftime('%Y-%m-%d'), "5d")
            if hist_s.empty:
                return jsonify({"error": f"Could not fetch stock price for {ticker_str} around {quote_date_str}."}), 400
            S = hist_s['Close'].iloc[-1]
        else:
            S = hist_s['Close'].iloc[0]

        start_vol_date = quote_date - timedelta(days=365)
        hist_vol_start_str = (start_vol_date - timedelta(days=5)).strftime('%Y-%m-%d')
        hist_vol_end_str = quote_date.strftime('%Y-%m-%d')
        hist_vol = get_stock_history(ticker_str, hist_vol_start_str, hist_vol_end_str)

        if len(hist_vol) < 2:
            return jsonify({"error": f"Not enough historical data for {ticker_str} to calculate volatility before {quote_date_str}."}), 400

        hist_vol['LogReturn'] = np.log(hist_vol['Close'] / hist_vol['Close'].shift(1))
        daily_std_dev = hist_vol['LogReturn'].std()
        sigma = daily_std_dev * np.sqrt(252)

        if np.isnan(sigma) or sigma == 0:
            return jsonify({"error": f"Could not calculate valid volatility for {ticker_str}."}), 400

        T = (expiry_date - quote_date).days / 365.0
        if T <= 0:
            return jsonify({"error": "Time to maturity must be positive."}), 400

        r = 0.05
        try:
            hist_r_end_date_str = (quote_date + timedelta(days=1)).strftime('%Y-%m-%d')
            hist_r = get_risk_free_rate_history(hist_r_end_date_str, "5d")
            if not hist_r.empty and 'Close' in hist_r.columns:
                r_percentage = hist_r['Close'].iloc[-1]
                r = r_percentage / 100.0
            else:
                app.logger.warning(f"Could not fetch risk-free rate (^IRX) around {quote_date_str}. Using default 0.")
        except Exception as r_err:
            app.logger.error(f"Error fetching risk-free rate (^IRX): {r_err}")
            r = 0.05
            app.logger.warning(f"Using fallback risk-free rate: {r}")

        american = (exercise_style == 'american')

        plot_data = {
            "S": S,
            "K": K,
            "T": T,
            "r": r,
            "sigma": sigma,
            "option_type": option_type,
            "exercise_style": exercise_style,
            "n_simulations": num_paths_analytics, 
            "n_steps": num_steps_analytics
        }

        price = None
        plot_json = None
        analytics_package = {}
        plot_result = {}

        if model == "black_scholes":
            plot_result = plot_black_scholes(plot_data)
            price = plot_result.get("price")
            plot_json = plot_result.get("plot")
            analytics_package = generate_full_analytics_package(plot_data, price, "Black-Scholes", num_paths_sim=num_paths_analytics, num_steps_sim=num_steps_analytics)
        elif model == "binomial":
            plot_data["n_steps"] = data.get("n_steps", 100)
            plot_result = plot_binomial(plot_data)
            price = plot_result.get("price")
            plot_json = plot_result.get("plot")
            analytics_package = generate_full_analytics_package(plot_data, price, "Binomial", num_paths_sim=num_paths_analytics, num_steps_sim=plot_data["n_steps"])
        elif model == "monte_carlo":
            n_simulations = data.get("n_simulations", 10000)
            n_steps = data.get("n_steps", 100)
            plot_data["n_simulations"] = n_simulations
            plot_data["n_steps"] = n_steps
            plot_result = plot_monte_carlo(plot_data)
            price = plot_result.get("price")
            plot_json = plot_result.get("plot")
        elif model == "pde":
            if american:
                app.logger.warning("PDE model requested with American style, forcing European.")
                plot_data["exercise_style"] = 'european'
            plot_data["n_t"] = data.get("n_t", 1000)
            plot_data["x_max"] = data.get("x_max", S * 2)
            plot_result = plot_pde(plot_data)
            price = plot_result.get("price")
            plot_json = plot_result.get("plot")
            analytics_package = generate_full_analytics_package(plot_data, price, "PDE", num_paths_sim=num_paths_analytics, num_steps_sim=num_steps_analytics)
        else:
            return jsonify({"error": f"Model '{model}' not supported for historical pricing/plotting yet."}), 400

        final_response = {
            "S": S,
            "sigma": sigma,
            "T": T,
            "r": r,
            "price": price,
            "plot": plot_json,
        }
        final_response.update(analytics_package)
        return jsonify(final_response)

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


def plot_gbm_simulation(S0, K_strike, T, r, sigma, steps=100, num_paths=100):
    time_array, paths = simulate_gbm_paths(S0, T, r, sigma, steps, num_paths)
    fig = go.Figure()
    fig.layout.template = 'plotly_white'

    if isinstance(paths, np.ndarray) and paths.ndim == 2:
        for i in range(paths.shape[1]):
            fig.add_trace(go.Scatter(
                x=time_array.tolist(),
                y=paths[:, i].tolist(),
                mode='lines',
                showlegend=False,
                line=dict(width=1)
            ))
    else:
        app.logger.error(f"GBM paths not in expected format. Type: {type(paths)}, Shape: {getattr(paths, 'shape', 'N/A')}")

    fig.add_shape(
        type="line",
        x0=0,
        x1=T,
        y0=K_strike,
        y1=K_strike,
        line=dict(color="red", width=2, dash="dash"),
        name="Strike Price"
    )
    fig.update_layout(title=f"GBM Stock Price Simulations (K={K_strike:.2f})",
                      xaxis_title='Time (Years)',
                      yaxis_title='Stock Price')
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


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
    K = float(data["K"])
    T = data["T"]
    r = data["r"]
    sigma = data["sigma"]
    x_max_default = 200 if data.get("option_type", "call") == 'call' else 3.0
    x_max = data.get("x_max", x_max_default)
    n_t = data.get("n_t", 1000)

    option_type = data.get("option_type", "call")

    if option_type == "call":
        x, V, user_price = crank_nicolson_call(S_user, K, sigma, T, r, x_max=x_max, N_t=n_t)
    else:
        x, V, user_price = crank_nicolson_put(S_user, K, sigma, T, x_max=x_max, N_t=n_t)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=V, mode='lines', name='CN Price'))
    fig.add_trace(go.Scatter(x=[S_user], y=[user_price],
                             mode='markers', marker=dict(color='red', size=10), name='Current S'))
    fig.update_layout(title='Crank–Nicolson Option Price',
                      xaxis_title='Stock Price (S)',
                      yaxis_title='Option Price')

    return {
        "plot": json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder),
        "price": user_price
    }


def plot_monte_carlo(data):
    S_user = data["S"]
    K = float(data["K"])
    T = data["T"]
    r = data["r"]
    sigma = data["sigma"]
    option_type = data.get("option_type", "call")
    exercise_style = data.get("exercise_style", "european")
    n_simulations = data.get("n_simulations", 10000)
    n_steps = data.get("n_steps", 100)

    s_min = max(5, int(S_user * 0.5))
    s_max = int(S_user * 1.5)
    step = max(1, int((s_max - s_min) / 20))
    S_range = list(range(s_min, s_max + step, step))

    prices = []
    for S_val in S_range:
        pricer = MonteCarloOptionPricer(
            spot_price=S_val,
            strike_price=K,
            time_to_maturity=T,
            risk_free_rate=r,
            volatility=sigma,
            option_type=option_type,
            num_simulations=n_simulations,
            num_steps=n_steps
        )
        
        if exercise_style == 'american':
            price, _ = pricer.price_american_option()
        else:
            price, _ = pricer.price_european_option()
        prices.append(price)

    # Calculate price for user's spot price
    user_pricer = MonteCarloOptionPricer(
        spot_price=S_user,
        strike_price=K,
        time_to_maturity=T,
        risk_free_rate=r,
        volatility=sigma,
        option_type=option_type,
        num_simulations=n_simulations,
        num_steps=n_steps
    )
    
    if exercise_style == 'american':
        user_price, _ = user_pricer.price_american_option()
    else:
        user_price, _ = user_pricer.price_european_option()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S_range, y=prices, mode='lines+markers', name='Monte Carlo Price'))
    fig.add_trace(go.Scatter(x=[S_user], y=[user_price],
                             mode='markers', marker=dict(color='red', size=10), name='Current S'))
    fig.update_layout(title='Monte Carlo Option Price',
                      xaxis_title='Stock Price (S)',
                      yaxis_title='Option Price')

    return {
        "plot": json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder),
        "price": user_price
    }


if __name__ == "__main__":
    app.run(debug=True)


