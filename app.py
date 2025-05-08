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
from models.gbm import simulate_gbm_paths, calculate_payoffs, get_gbm_analytics


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


def plot_terminal_prices_histogram(terminal_prices_counts, terminal_prices_bin_edges, K_strike):
    """Generates a Plotly JSON for the terminal stock prices histogram."""
    fig = go.Figure(data=[go.Bar(
        x=[(terminal_prices_bin_edges[i] + terminal_prices_bin_edges[i+1])/2 for i in range(len(terminal_prices_counts))],  # Bin centers
        y=terminal_prices_counts,
        marker_color='#2ca02c' # Green color for stock prices
    )])
    fig.add_vline(x=K_strike, line_width=2, line_dash="dash", line_color="red", name="Strike Price")
    fig.update_layout(
        title=f"Distribution of Terminal Stock Prices (S<sub>T</sub>)",
        xaxis_title="Terminal Stock Price (S<sub>T</sub>) ($)",
        yaxis_title="Frequency (Number of Simulations)",
        bargap=0.1,
        template='plotly_white',
        shapes=[dict(
            type='line',
            yref='paper', y0=0, y1=1,
            xref='x', x0=K_strike, x1=K_strike,
            line=dict(color='red', width=2, dash='dash')
        )],
        annotations=[dict(
            x=K_strike, y=max(terminal_prices_counts) * 0.95 if terminal_prices_counts else 0, # Position annotation near strike
            xref='x', yref='y',
            text=f'Strike K={K_strike:.2f}', showarrow=True, arrowhead=1, ax=20, ay=-30
        )],
        autosize=True # Ensure autosize
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def generate_full_analytics_package(data, model_price, model_name_str, num_paths_sim=1000, num_steps_sim=100):
    """
    Helper to generate GBM simulation, its plot, MC price, Terminal Prices histogram, and stats.
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

    # 4. P&L analysis and Summary Stats (and Terminal Price histogram data)
    # Use the primary model's price as the option_cost for P&L
    summary_stats, terminal_prices_counts, terminal_prices_bin_edges = get_gbm_analytics(
        terminal_prices_analytics, 
        payoffs_analytics, 
        K, T, r, option_type, 
        option_cost=model_price 
    )
    
    # 5. Terminal Prices Histogram Plot
    terminal_prices_histogram_plot_json = plot_terminal_prices_histogram(terminal_prices_counts, terminal_prices_bin_edges, K)

    return {
        "gbm_simulation_plot": gbm_simulation_plot_json,
        "monte_carlo_price_from_analytics_sim": mc_price_sim,
        "gbm_summary_stats": summary_stats,
        "terminal_prices_histogram_plot": terminal_prices_histogram_plot_json
    }


def plot_all_models_data(data, market_price_from_payload):
    """
    Generates a combined plot, pricing table, and error bar chart for all models.
    """
    S_user = data["S"]
    K = float(data["K"])
    T = data["T"]
    r = data["r"]
    sigma = data["sigma"]
    option_type = data["option_type"]
    exercise_style = data.get("exercise_style", "european")
    american = (exercise_style == 'american')

    # Parameters for models that need them (e.g., Binomial, MC)
    n_steps = data.get("n_steps", 100)  # Default if not provided
    n_simulations = data.get("n_simulations", 10000) # Default for MC

    # Calculate prices for all models
    bs_price = price_option(S_user, K, T, r, sigma, option_type)
    print("Black-Scholes price:", bs_price)
    binomial_price_val = binomial_price(S_user, K, T, r, sigma, n_steps=n_steps, option_type=option_type, american=american)
    print("Binomial price:", binomial_price_val)
    # Monte Carlo Simulation
    mc_time_array, mc_price_paths = simulate_gbm_paths(S_user, T, r, sigma, n_steps, n_simulations) # n_steps for MC time steps
    mc_terminal_prices = mc_price_paths[-1, :]
    mc_payoffs = calculate_payoffs(mc_terminal_prices, K, option_type)
    mc_discounted_payoffs = np.exp(-r * T) * mc_payoffs
    mc_price = np.mean(mc_discounted_payoffs)
    mc_error = np.std(mc_discounted_payoffs) / np.sqrt(n_simulations)
    print("Finished MC simulation. Price:", mc_price, "Error:", mc_error)

    # PDE Price (Crank-Nicolson) - Assuming European for simplicity in "All Models" view
    # PDE model here doesn't easily take 'american' for calls.
    # Using default x_max and N_t for PDE in this combined view.
    x_max_pde = data.get("x_max", 200 if option_type == 'call' else S_user * 2) # Sensible defaults
    n_t_pde = data.get("n_t", 1000)
    if option_type == "call":
        _, _, pde_price = crank_nicolson_call(S_user, K, sigma, T, r, x_max=x_max_pde, N_t=n_t_pde)
    else: # put
        _, _, pde_price = crank_nicolson_put(S_user, K, sigma, T, x_max=x_max_pde, N_t=n_t_pde)
    print("PDE price:", pde_price)

    # Generate bar chart for model prices
    model_names = ["Black-Scholes", "Binomial", "Monte Carlo", "PDE"]
    model_prices = [bs_price, binomial_price_val, mc_price, pde_price]

    fig_bar_prices = go.Figure(data=[go.Bar(
        x=model_names,
        y=model_prices,
        text=[f"${p:.2f}" for p in model_prices],
        textposition='auto',
        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Colors for bars
    )])
    fig_bar_prices.update_layout(
        title="Option Prices by Model",
        xaxis_title="Model",
        yaxis_title="Option Price ($)"
    )
    combined_plot_json = json.dumps(fig_bar_prices, cls=plotly.utils.PlotlyJSONEncoder)

    # Generate pricing table data
    pricing_table = [
        {"Model": "Black-Scholes", "Price": f"${bs_price:.2f}"},
        {"Model": "Binomial", "Price": f"${binomial_price_val:.2f}"},
        {"Model": "Monte Carlo", "Price": f"${mc_price:.2f} ± {mc_error:.2f}"},
        {"Model": "PDE", "Price": f"${pde_price:.2f}"},
    ]
    if market_price_from_payload is not None:
        pricing_table.append({"Model": "Market Price", "Price": f"${market_price_from_payload:.2f}"})

    # Generate pricing error bar chart (only if market_price was provided)
    error_bar_chart_json = None
    if market_price_from_payload is not None:
        models_for_error = ["Black-Scholes", "Binomial", "Monte Carlo", "PDE"]
        prices_for_error = [bs_price, binomial_price_val, mc_price, pde_price]
        errors = [abs(p - market_price_from_payload) for p in prices_for_error]
        
        fig_error = go.Figure(data=[go.Bar(x=models_for_error, y=errors, name='Absolute Error')])
        fig_error.update_layout(title="Absolute Pricing Error vs Market Price", xaxis_title="Model", yaxis_title="Absolute Error ($)")
        error_bar_chart_json = json.dumps(fig_error, cls=plotly.utils.PlotlyJSONEncoder)
    print("Pricing table:", pricing_table)
    return {
        "combined_plot": combined_plot_json,
        "pricing_table": pricing_table,
        "error_bar_chart": error_bar_chart_json
    }


@app.route("/plot", methods=["POST"])
def route_plot():
    data = request.json
    model = data.get("model", "black_scholes")
    market_price = data.get("market_price") # This will be a float or None (if JS sent null)
    vol_source = data.get("volatility_source", "constant").lower() # Added

    if vol_source == "lstm": # Check for LSTM in manual mode
        return jsonify({"error": "LSTM volatility is only supported for historical data."}), 400
    if vol_source == "garch": # Check for GARCH in manual mode
        return jsonify({"error": "GARCH volatility is only supported for historical data."}), 400

    if "K" in data:
        data["K"] = float(data["K"])
    
    # Default values for analytics simulations
    num_paths_analytics = data.get("n_simulations", 1000)  # Use n_simulations if provided, else 1000
    num_steps_analytics = data.get("n_steps", 100)  # Use n_steps if provided, else 100

    if model == "all":
        result = plot_all_models_data(data, market_price) # Pass market_price here
        return jsonify(result)

    if model == "black_scholes":
        result = plot_black_scholes(data)
        return jsonify(result)

    elif model == "binomial":
        result = plot_binomial(data)
        return jsonify(result)

    elif model == "monte_carlo":
        bs_ref_price = price_option(data["S"], float(data["K"]), data["T"], data["r"], data["sigma"], data["option_type"])
        analytics_package = generate_full_analytics_package(data, bs_ref_price, "Monte Carlo (GBM Analytics)", num_paths_sim=num_paths_analytics, num_steps_sim=num_steps_analytics)
        
        mc_price_from_sim = analytics_package.get("monte_carlo_price_from_analytics_sim")
        if isinstance(mc_price_from_sim, str):
            try:
                mc_price_value = float(mc_price_from_sim.replace("$",""))
            except ValueError:
                mc_price_value = None
        else:
            mc_price_value = mc_price_from_sim

        return jsonify({
            "price": mc_price_value,
            "plot": None,
            "gbm_simulation_plot": analytics_package.get("gbm_simulation_plot"),
            "gbm_summary_stats": analytics_package.get("gbm_summary_stats"),
            "terminal_prices_histogram_plot": analytics_package.get("terminal_prices_histogram_plot"),
            "monte_carlo_price_from_analytics_sim": mc_price_value
        })

    elif model == "pde":
        result = plot_pde(data)
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
    vol_source = data.get("volatility_source", "constant").lower() # Added
    
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
        log_returns = hist_vol['LogReturn'].dropna()

        if log_returns.empty:
            return jsonify({"error": f"Could not calculate log returns for {ticker_str}."}), 400

        # Volatility calculation based on source
        sigma_display_name = "Constant (Historical)" # Default display name
        if vol_source == "lstm":
            try:
                from lstm_helper import get_lstm_volatility
                sigma = get_lstm_volatility(log_returns, scale_factor=np.sqrt(252))
                sigma_display_name = "LSTM Forecast"
            except FileNotFoundError as fnf_err:
                app.logger.error(f"LSTM helper error: {fnf_err}")
                return jsonify({"error": str(fnf_err)}), 500
            except Exception as lstm_err:
                app.logger.error(f"LSTM volatility calculation failed: {lstm_err}")
                # Fallback to constant historical volatility on LSTM error
                sigma = log_returns.std() * np.sqrt(252)
                sigma_display_name = f"LSTM Error - Fallback Constant ({sigma:.4f})"
                # Optionally, return an error to the user:
                # return jsonify({"error": f"LSTM volatility calculation error: {str(lstm_err)}"}), 500
        elif vol_source == "garch":
            try:
                from garch_helper import get_garch_volatility 
                sigma = get_garch_volatility(log_returns) # Assumes it returns annualized vol
                sigma_display_name = "GARCH"
            except ImportError:
                app.logger.error("GARCH helper not found, falling back to constant volatility.")
                sigma = log_returns.std() * np.sqrt(252)
                sigma_display_name = f"GARCH Error - Fallback Constant ({sigma:.4f})"
                # return jsonify({"error": "GARCH model processing failed (helper not found). Please select another volatility source."}), 500
            except Exception as e_garch:
                app.logger.error(f"GARCH volatility calculation failed: {e_garch}")
                sigma = log_returns.std() * np.sqrt(252)
                sigma_display_name = f"GARCH Error - Fallback Constant ({sigma:.4f})"
                # return jsonify({"error": f"GARCH volatility calculation error: {str(e_garch)}"}), 500
        else:  # default to "constant" vol
            sigma = log_returns.std() * np.sqrt(252)
            # sigma_display_name is already "Constant (Historical)"

        if np.isnan(sigma) or sigma <= 0: # Sigma must be positive
            return jsonify({"error": f"Could not calculate valid positive volatility for {ticker_str} (source: {vol_source}, calculated value: {sigma})."}), 400

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

        if model == "black_scholes":
            plot_result = plot_black_scholes(plot_data)
            price = plot_result.get("price")
            plot_json = plot_result.get("plot")
        elif model == "binomial":
            plot_result = plot_binomial(plot_data)
            price = plot_result.get("price")
            plot_json = plot_result.get("plot")
        elif model == "monte_carlo":
            bs_ref_price = price_option(S, K, T, r, sigma, option_type)
            analytics_package = generate_full_analytics_package(plot_data, bs_ref_price, "Monte Carlo (GBM Analytics)", num_paths_sim=num_paths_analytics, num_steps_sim=num_steps_analytics)
            
            mc_price_from_sim_hist = analytics_package.get("monte_carlo_price_from_analytics_sim")
            if isinstance(mc_price_from_sim_hist, str):
                try:
                    price = float(mc_price_from_sim_hist.replace("$",""))
                except ValueError:
                    price = None 
            else:
                price = mc_price_from_sim_hist

            plot_json = None
        elif model == "pde":
            if american:
                app.logger.warning("PDE model requested with American style for historical, forcing European.")
                plot_data["exercise_style"] = 'european'
            plot_result = plot_pde(plot_data)
            price = plot_result.get("price")
            plot_json = plot_result.get("plot")
        else:
            return jsonify({"error": f"Model '{model}' not supported for historical pricing/plotting yet."}), 400

        final_response = {
            "S": S,
            "sigma": sigma,
            "T": T,
            "r": r,
            "price": price,
            "plot": plot_json,
            "volatility_source": sigma_display_name # Added for frontend display
        }
        if model == "monte_carlo":
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
    fig.update_layout(
        title='Black-Scholes Option Price',
        xaxis_title='Stock Price (S)',
        yaxis_title='Option Price',
        autosize=True,
    )

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
                      yaxis_title='Stock Price',
                      autosize=True) # Ensure autosize
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
    fig.update_layout(
        title='Binomial Option Price',
        xaxis_title='Stock Price (S)',
        yaxis_title='Option Price',
        autosize=True,
    )

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
    x_max_default = 200 if data.get("option_type", "call") == 'call' else S_user * 2 # Adjusted default for put
    x_max = data.get("x_max", x_max_default)
    n_t = data.get("n_t", 1000)

    option_type = data.get("option_type", "call")

    if option_type == "call":
        x, V, user_price = crank_nicolson_call(S_user, K, sigma, T, r, x_max=x_max, N_t=n_t)
    else:
        x, V, user_price = crank_nicolson_put(S_user, K, sigma, T, r, x_max=x_max, N_t=n_t) # Added r for put

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=V, mode='lines', name='CN Price'))
    fig.add_trace(go.Scatter(x=[S_user], y=[user_price],
                             mode='markers', marker=dict(color='red', size=10), name='Current S'))
    fig.update_layout(
        title='Crank–Nicolson Option Price',
        xaxis_title='Stock Price (S)',
        yaxis_title='Option Price',
        autosize=True,
    )

    return {
        "plot": json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder),
        "price": user_price
    }


if __name__ == "__main__":
    app.run(debug=True)



