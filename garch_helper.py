# --- Imports ---
import pandas as pd
import numpy as np
import yfinance as yf
from arch import arch_model
import matplotlib.pyplot as plt
import datetime
import warnings
# --- GARCH Volatility Calculation ---
def get_garch_volatility(log_returns: pd.Series) -> float:
    """
    Calculates GARCH(1,1) forecasted volatility.
    1. Fit a GARCH(1,1) model to the log_returns.
    2. Forecast the next-step conditional volatility.
    3. Annualize the forecasted volatility.
    """
    if log_returns.empty or len(log_returns) < 20:
        print("GARCH Warning: Insufficient data for GARCH. Returning historical volatility.")
        return log_returns.std() * np.sqrt(252) if not log_returns.empty else 0.20

    try:
        cleaned_returns = log_returns.dropna()
        if cleaned_returns.empty or len(cleaned_returns) < 20:
            print("GARCH Warning: Insufficient data after cleaning for GARCH. Returning historical volatility.")
            return log_returns.std() * np.sqrt(252) if not log_returns.empty else 0.20

        # Define GARCH(1,1) model
        model = arch_model(cleaned_returns, vol='Garch', p=1, q=1, mean='Constant', rescale=False)
        results = model.fit(disp='off', show_warning=False)
        forecast = results.forecast(horizon=1, reindex=False) # reindex=False for simpler access
        forecasted_variance_daily = forecast.variance.iloc[-1, 0]
        
        # Convert daily variance to annualized volatility
        # Volatility is sqrt of variance. Annualize by sqrt(252).
        annualized_vol = np.sqrt(forecasted_variance_daily) * np.sqrt(252)
        
        if np.isnan(annualized_vol) or annualized_vol <= 0:
            print(f"GARCH Warning: Invalid volatility forecasted ({annualized_vol}). Returning historical.")
            return cleaned_returns.std() * np.sqrt(252)
            
        print(f"GARCH Annualized Volatility Forecast: {annualized_vol:.4f}")
        return annualized_vol

    except Exception as e:
        print(f"GARCH Model Error: {e}. Falling back to historical volatility.")
        return log_returns.dropna().std() * np.sqrt(252) if not log_returns.dropna().empty else 0.20

# --- Testing the Function ---
if __name__ == '__main__':
    np.random.seed(42)
    dummy_log_returns = pd.Series(np.random.normal(loc=0.0005, scale=0.015, size=500)) 
    
    print("\n--- Testing with dummy_log_returns ---")
    garch_vol = get_garch_volatility(dummy_log_returns)
    print(f"Calculated GARCH Annualized Volatility: {garch_vol:.4f}")

    print("\n--- Testing with insufficient data ---")
    insufficient_returns = pd.Series(np.random.normal(0, 0.01, 10))
    garch_vol_insufficient = get_garch_volatility(insufficient_returns)
    print(f"Calculated GARCH (insufficient data) Volatility: {garch_vol_insufficient:.4f}")

    print("\n--- Testing with empty data ---")
    empty_returns = pd.Series([], dtype=float)
    garch_vol_empty = get_garch_volatility(empty_returns)
    print(f"Calculated GARCH (empty data) Volatility: {garch_vol_empty:.4f}")

    print("\n--- Testing with data containing NaNs ---")
    nan_returns = pd.Series([0.01, 0.005, np.nan, -0.01, 0.02] * 20) # Ensure enough non-NaNs
    garch_vol_nan = get_garch_volatility(nan_returns)
    print(f"Calculated GARCH (NaN data) Volatility: {garch_vol_nan:.4f}")
