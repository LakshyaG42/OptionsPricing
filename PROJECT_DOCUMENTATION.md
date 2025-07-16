# Project Documentation: OptionsPricing

This document outlines the various functions, modules, and implementations within the OptionsPricing project.

## 1. Backend (Flask Application - `app.py`)

The core backend logic is handled by a Flask application.

### 1.1. Flask App Setup
-   **`app = Flask(__name__)`**: Initializes the Flask application.
-   **`Cache(app)`**: Sets up caching for the application, configured for `SimpleCache`.

### 1.2. Routing
-   **`@app.route("/")` (Function: `home`)**:
    -   Serves the main `index.html` page.
-   **`@app.route("/plot", methods=["POST"])` (Function: `route_plot`)**:
    -   Handles requests for option pricing and plotting in "Manual Input" mode.
    -   Receives JSON data with option parameters (S, K, T, r, sigma, option_type, model, exercise_style, market_price, etc.).
    -   Validates against LSTM/GARCH usage in manual mode.
    -   If `model == "all"`, calls `plot_all_models_data`.
    -   Otherwise, routes to specific model plotting functions (`plot_black_scholes`, `plot_binomial`, `plot_pde`) or generates Monte Carlo analytics via `generate_full_analytics_package`.
    -   Returns JSON response with plot data, price, and analytics.
-   **`@app.route("/historical_price", methods=["POST"])` (Function: `route_historical_price`)**:
    -   Handles requests for option pricing and plotting in "Real World Data" mode.
    -   Receives JSON data with ticker, quote date, expiry date, K, option type, model, exercise style, and volatility source.
    -   Fetches historical stock data (`get_stock_history`, `get_stock_history_period`) and risk-free rates (`get_risk_free_rate_history`).
    -   Calculates Time to Maturity (T).
    -   Calculates volatility (`sigma`) based on the `volatility_source`:
        -   `constant`: Standard deviation of log returns over 1 year.
        -   `short_term_historical`: Standard deviation of log returns over 1 month (21 trading days).
        -   `garch`: Calls `garch_helper.get_garch_volatility`.
        -   `lstm`: Calls `lstm_helper.get_lstm_volatility`.
    -   Prepares data for the volatility chart.
    -   Calls the appropriate model pricing/plotting function similar to `/plot`.
    -   Returns JSON response with calculated parameters (S, sigma, T, r), price, plot data, volatility source display name, and volatility chart data.

### 1.3. Data Fetching and Caching
-   **`@cache.memoize(timeout=...)`**: Decorator used to cache results of data fetching functions to reduce redundant API calls.
-   **`get_stock_history(ticker_symbol, start_date_str, end_date_str)`**: Fetches stock history for a given ticker between specified dates using `yfinance`.
-   **`get_stock_history_period(ticker_symbol, end_date_str, period_str)`**: Fetches stock history for a given ticker for a specified period ending on `end_date_str`.
-   **`get_risk_free_rate_history(end_date_str, period_str)`**: Fetches historical data for the 13-week Treasury Bill (^IRX) to use as a proxy for the risk-free rate.

### 1.4. Plotting Helper Functions (Plotly)
-   **`plot_pnl_histogram(...)`**: *Currently not directly used in the main flow but available.* Generates Plotly JSON for a P&L histogram.
-   **`plot_terminal_prices_histogram(terminal_prices_counts, terminal_prices_bin_edges, K_strike)`**: Generates Plotly JSON for a histogram of terminal stock prices from Monte Carlo simulations, including a line for the strike price.
-   **`plot_gbm_simulation(S0, K_strike, T, r, sigma, steps=100, num_paths=100)`**: Generates Plotly JSON for a plot of Geometric Brownian Motion (GBM) paths.
-   **`plot_black_scholes(data)`**: Calculates Black-Scholes prices over a range of stock prices (S_range) and generates a Plotly line chart. Returns the plot JSON and the price at the user-specified S.
-   **`plot_binomial(data)`**: Calculates Binomial Tree prices over S_range and generates a Plotly line chart. Returns plot JSON and price at user S.
-   **`plot_pde(data)`**: Calculates Crank-Nicolson PDE prices over S_range and generates a Plotly line chart. Returns plot JSON and price at user S.

### 1.5. Analytics and Model Aggregation
-   **`generate_full_analytics_package(data, model_price, model_name_str, num_paths_sim=1000, num_steps_sim=100)`**:
    -   Orchestrates Monte Carlo simulation analytics.
    -   Calls `simulate_gbm_paths` for path generation.
    -   Calls `plot_gbm_simulation` for the path plot.
    -   Calculates payoffs using `calculate_payoffs`.
    -   Calculates Monte Carlo price from simulations.
    -   Calls `get_gbm_analytics` for summary statistics and terminal price histogram data.
    -   Calls `plot_terminal_prices_histogram` for the histogram plot.
    -   Returns a dictionary containing all these analytics components.
-   **`plot_all_models_data(data, market_price_from_payload)`**:
    -   Calculates option prices using Black-Scholes, Binomial, Monte Carlo, and PDE models.
    -   Generates a Plotly bar chart comparing these model prices.
    -   Creates a pricing table (list of dictionaries) for display.
    -   If `market_price_from_payload` is provided, generates an error bar chart showing the absolute difference between model prices and the market price.
    -   Returns a dictionary with the combined plot, pricing table, and error bar chart.

## 2. Pricing Models (`models/`)

### 2.1. Black-Scholes (`black_scholes.py`)
-   **`price_option(S, K, T, r, sigma, option_type="call")`**:
    -   Calculates the Black-Scholes option price for a European call or put.
    -   Uses `scipy.stats.norm.cdf` for the cumulative standard normal distribution.

### 2.2. Binomial Tree (`binomial.py`)
-   **`binomial_price(S, K, T, r, sigma, n_steps=100, option_type="call", american=False)`**:
    -   Implements the Cox-Ross-Rubinstein (CRR) binomial tree model.
    -   Calculates option price for European or American style calls/puts.
    -   Uses backward induction to determine the price at t=0.

### 2.3. Monte Carlo (GBM) (`gbm.py`)
-   **`simulate_gbm_paths(S0, T, r, sigma, steps, num_paths)`**:
    -   Simulates multiple stock price paths using Geometric Brownian Motion.
    -   Returns a time array and a 2D NumPy array of price paths.
-   **`calculate_payoffs(terminal_prices, K, option_type)`**:
    -   Calculates the payoff for call or put options given terminal prices and strike.
-   **`monte_carlo_option_price(S0, K, T, r, sigma, option_type, steps=100, num_paths=1000)`**:
    -   A standalone function to price an option using Monte Carlo simulation.
    -   Combines `simulate_gbm_paths` and `calculate_payoffs`.
    -   Returns the MC price, terminal prices, and payoffs.
-   **`get_gbm_analytics(terminal_prices, payoffs, K, T, r, option_type, option_cost)`**:
    -   Calculates various statistics from GBM simulation results, including:
        -   Monte Carlo price (discounted expected payoff).
        -   Expected payoff.
        -   Expected P&L (Profit and Loss) against a given `option_cost`.
        -   Probability of profit.
        -   Value at Risk (5th percentile of P&L).
        -   Average and median final stock price.
        -   Probability of the option expiring in-the-money (ITM).
    -   Generates data (counts and bin edges) for the terminal prices histogram.
    -   Returns a dictionary of formatted statistics and histogram data.

### 2.4. PDE (Crank-Nicolson) (`pde.py`)
-   **`norm_cdf(x)`**: Helper for CDF, using `math.erf`.
-   **`bs_put_price(S, K, sigma, T)`**: Black-Scholes put price, used for boundary conditions.
-   **`solve_tridiagonal_scipy(a, b, c, d)`**: Solves a tridiagonal matrix system using `scipy.linalg.solve_banded`. This is the core solver for the implicit steps in Crank-Nicolson.
-   **`crank_nicolson_put(S, K, sigma, T, r, x_max=3.0, N_t=1000)`**:
    -   Prices a European put option using the Crank-Nicolson finite difference method to solve the Black-Scholes PDE.
    -   Sets up the grid (space `x` and time `t`).
    -   Applies terminal and boundary conditions for a put option.
    -   Iteratively solves backwards in time using `solve_tridiagonal_scipy`.
    -   Interpolates the price at the initial stock price `S`.
    -   Returns the grid `x`, final option values `V` across the grid, and the interpolated price `V[idx]`.
-   **`crank_nicolson_call(S, K, sigma, T, r, x_max=200, N_t=1000)`**:
    -   Similar to `crank_nicolson_put` but for a European call option, with appropriate terminal and boundary conditions.

## 3. Volatility Helpers

### 3.1. GARCH (`garch_helper.py`)
-   **`get_garch_volatility(log_returns: pd.Series) -> float`**:
    -   Fits a GARCH(1,1) model to the provided log returns using the `arch` library.
    -   Forecasts the 1-day ahead conditional variance.
    -   Converts the daily variance to annualized volatility (sqrt(variance) * sqrt(252)).
    -   Includes error handling and fallbacks to historical volatility if GARCH fitting fails or data is insufficient.

### 3.2. LSTM (`lstm_helper.py`)
-   **`LSTMModel(nn.Module)`**:
    -   Defines the PyTorch LSTM model architecture (LSTM layers, Dropout, Linear output layer).
-   **`create_sequences(data, window_size)`**:
    -   Prepares input sequences for the LSTM model from time series data. For prediction, it takes the most recent `window_size` data points.
-   **`load_lstm_model_once()`**:
    -   Loads the pre-trained LSTM model (`lstm_volatility_model.pth`) and scaler.
    -   Ensures the model is loaded only once.
    -   Handles model loading to the appropriate device (CPU/CUDA).
-   **`get_lstm_volatility(pred_input: pd.Series, scale_factor: float = 1.0) -> float`**:
    -   Calculates realized volatility from the input `pred_input` log returns.
    -   Scales the realized volatility using the pre-fitted `StandardScaler` (Note: in the current implementation, the scaler is re-fit on current input as a simplification; ideally, it should be saved from training).
    -   Creates an input sequence using `create_sequences`.
    -   Uses the loaded LSTM model to predict the next-step scaled volatility.
    -   Inverse-transforms the prediction to get the actual volatility.
    -   Returns the forecasted annualized volatility.
    -   Includes error handling and fallbacks.

## 4. Frontend (JavaScript - `static/script.js`)

Handles user interactions, form submissions, and dynamic plot rendering on the client-side.

### 4.1. Global State and Helpers
-   **`modelParamMap`**: Object mapping model names to their specific input parameter fields (e.g., `n_steps` for Binomial).
-   **`lastSuccessfulPayload`, `lastSuccessfulHistoricalPayload`**: Variables to cache the last successful form submission data to avoid redundant API calls if inputs haven't changed.
-   **`deepEqual(obj1, obj2)`**: Helper function to compare two objects deeply.

### 4.2. DOM Element References
-   Variables storing references to key HTML elements (forms, input fields, plot divs, buttons, display areas).

### 4.3. Mode Switching
-   **`handleModeChange()`**:
    -   Triggered when the user switches between "Manual Input" and "Real World Data" modes.
    -   Shows/hides the relevant form sections.
    -   Clears all plots and result displays (`Plotly.purge`, hiding divs, clearing text).
    -   Updates visibility of model-specific inputs for the newly active mode.

### 4.4. Dynamic Input Visibility
-   **`updateVisibleInputs()`**:
    -   Manages the visibility of input fields specific to the selected model in "Manual Input" mode (e.g., `n_steps` for Binomial, `n_simulations` for Monte Carlo).
    -   Hides the "American" exercise style option if Black-Scholes or PDE model is selected (as they primarily handle European in this implementation).
    -   Controls visibility of the "Rerun Simulations" button for Monte Carlo.
-   **`updateHistoricalVisibleInputs()`**:
    -   Similar to `updateVisibleInputs` but for the "Real World Data" mode.
    -   Hides "American" for Black-Scholes/PDE.
    -   Controls visibility of the "Rerun Simulations" button for historical Monte Carlo.

### 4.5. Date Handling
-   Sets default values and constraints for `quote_date` and `expiry_date` inputs (e.g., `quote_date` defaults to yesterday, `expiry_date` to one year from today).

### 4.6. Form Submission
-   **`handleManualFormSubmit()`**:
    -   Asynchronously handles submission of the "Manual Input" form.
    -   Collects input values into a `payload` object.
    -   Checks against `lastSuccessfulPayload` to use cached results if no change.
    -   Clears previous plots and results.
    -   Sends a POST request to `/plot` with the payload.
    -   Processes the JSON response:
        -   If `model == "all"`: Displays combined plot, pricing table, and error bar chart.
        -   If `model == "monte_carlo"`: Displays GBM simulation plot, terminal prices histogram, and summary statistics.
        -   For other models: Displays the main option price plot.
    -   Updates the output text with the calculated price(s).
    -   Handles errors and updates UI accordingly.
    -   Manages visibility of the "Rerun Simulations" button.
-   **`handleHistoricalFormSubmit()`**:
    -   Asynchronously handles submission of the "Real World Data" form.
    -   Collects input values (ticker, dates, K, model, etc.) into a `payload`.
    -   Checks against `lastSuccessfulHistoricalPayload`.
    -   Clears previous plots and results.
    -   Sends a POST request to `/historical_price`.
    -   Processes the JSON response:
        -   Displays calculated parameters (S, sigma, T, r, vol source).
        -   Plots the historical and forecasted volatility chart using `plotVolatilityChart`.
        -   If `model == "monte_carlo"`: Displays MC analytics similar to manual mode.
        -   For other models: Displays the main option price plot.
    -   Updates output text and handles errors.
    -   Manages visibility of the "Rerun Simulations" button.

### 4.7. Plot Rendering (Plotly)
-   Uses `Plotly.newPlot()` to render plots received from the backend.
-   Uses `Plotly.purge()` to clear plot divs.
-   **`plotVolatilityChart(volData)`**:
    -   Specifically handles rendering the historical and forecasted volatility chart.
    -   Takes `volData` (containing historical dates/values, quote date, expiry date, predicted sigma) from the backend.
    -   Creates traces for historical volatility and the forecasted volatility line/segment.
    -   Adds a vertical line and annotation for the quote date.
    -   Displays the chart in the `volatility_plot_div`.

### 4.8. Event Listeners
-   Listeners for mode changes (`modeRadios`).
-   Listeners for model selection changes (`model_manual`, `model_historical`) to trigger `updateVisibleInputs` or `updateHistoricalVisibleInputs`.
-   Listeners for form submissions (`optionForm`, `historicalForm`).
-   Listener for the "Rerun Simulations" button (`rerunButton`) to re-trigger form submission for Monte Carlo models without changing inputs (effectively re-simulating).
-   `DOMContentLoaded` listener for initial setup (date defaults, input visibility).

### 4.9. Clearing Results
-   **`clearHistoricalResults()`**: Clears displayed historical parameters and the volatility plot.

## 5. Jupyter Notebooks

### 5.1. `garch.ipynb`
-   **Purpose**: Development, testing, and demonstration of GARCH model fitting and volatility forecasting.
-   **Key Steps**:
    1.  Fetches historical stock data (e.g., AAPL).
    2.  Calculates log returns.
    3.  Fits a GARCH(1,1) model to the log returns.
    4.  Forecasts 1-day ahead volatility.
    5.  Visualizes log returns and the GARCH conditional volatility.

### 5.2. `lstm.ipynb`
-   **Purpose**: Development, training, and evaluation of an LSTM neural network for volatility forecasting.
-   **Key Steps**:
    1.  Loads historical stock data.
    2.  Calculates log returns and then realized volatility (target variable).
    3.  Scales the realized volatility.
    4.  Creates input sequences (X) and target values (y) for the LSTM.
    5.  Splits data into training and testing sets.
    6.  Sets up PyTorch DataLoaders.
    7.  Defines the LSTM model architecture (using `torch.nn.LSTM`, `nn.Linear`, `nn.Dropout`).
    8.  Trains the model using MSE loss and Adam optimizer, tracking training and test loss.
    9.  Evaluates the model on the test set (MSE, MAE on scaled and actual values).
    10. Plots true vs. predicted volatility.
    11. Provides an option to save the trained model's state dictionary (`lstm_volatility_model.pth`).

This documentation provides a high-level overview. For specific implementation details, refer to the comments within the respective source code files.
