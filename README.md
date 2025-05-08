# OptionsPricing

This project provides a web-based application for pricing various financial options and analyzing volatility. It allows users to price options using theoretical inputs or real-world historical data for selected stock tickers.

## Features

*   **Multiple Pricing Models**:
    *   Black-Scholes
    *   Binomial Tree (Cox-Ross-Rubinstein)
    *   Monte Carlo Simulation
    *   Partial Differential Equation (PDE) using Crank-Nicolson finite difference method
*   **Real-World Data Integration**:
    *   Fetches historical stock data using `yfinance`.
    *   Calculates historical volatility.
*   **Volatility Forecasting (Experimental)**:
    *   GARCH models for volatility prediction.
    *   LSTM-based models for volatility forecasting (see `lstm.ipynb`).
*   **Interactive Web Interface**:
    *   Built with Flask.
    *   Dynamic plotting of option price sensitivities, payoff diagrams, Monte Carlo simulation paths, and volatility forecasts using Plotly.
    *   Separate modes for manual input and historical data-driven pricing.
*   **Analytics**:
    *   For Monte Carlo simulations, provides GBM path visualizations, terminal price distributions, and P&L statistics.
    *   Comparison of prices from different models.

## Technologies Used

*   **Backend**: Python, Flask
*   **Frontend**: HTML, CSS, JavaScript
*   **Data Handling & Numerical Computation**: NumPy, Pandas, SciPy
*   **Financial Data**: yfinance
*   **Volatility Modeling**:
    *   `arch` (for GARCH models)
    *   `scikit-learn` (for data preprocessing)
    *   `statsmodels`
    *   `torch` (for LSTM models)
*   **Plotting**: Plotly
*   **Notebooks**: Jupyter Notebook (`.ipynb` files for model development and experimentation, e.g., `garch.ipynb`, `lstm.ipynb`)

## Setup and Running the Application

Follow these steps to set up and run the project locally:

1.  **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd OptionsPricing
    ```

2.  **Create and Activate a Virtual Environment**:
    It's highly recommended to use a virtual environment to manage project dependencies.

    *   **On macOS and Linux**:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

    *   **On Windows**:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    You should see `(venv)` at the beginning of your command prompt, indicating the virtual environment is active.

3.  **Install Dependencies**:
    With the virtual environment activated, install the required Python packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
    This will install Flask, NumPy, Pandas, yfinance, arch, Plotly, and other necessary libraries.

4.  **Run the Flask Application**:
    Once the dependencies are installed, you can run the Flask development server:
    ```bash
    python app.py
    ```
    The application will typically be available at `http://127.0.0.1:5000/` or `http://localhost:5000/` in your web browser. The console output will confirm the address.

5.  **Accessing the Application**:
    Open your web browser and navigate to the URL provided in the console (usually `http://127.0.0.1:5000/`).

## Notebooks

The repository includes Jupyter notebooks (`garch.ipynb`, `lstm.ipynb`) for developing and testing the volatility forecasting models. To run these, ensure you have Jupyter Notebook or JupyterLab installed in your virtual environment:
```bash
pip install jupyterlab notebook
```
Then, you can launch JupyterLab:
```bash
jupyter lab
```
Or Jupyter Notebook:
```bash
jupyter notebook
```
And navigate to the notebook files.

## To Deactivate the Virtual Environment (when done):
```bash
deactivate
```