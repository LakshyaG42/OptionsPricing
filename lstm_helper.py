# --- Imports ---
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import yfinance as yf
import os

# --- LSTM Model Definition ---
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, dropout_prob=0.25):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = self.linear(out)
        return out.squeeze()

# --- Sequence Creation Helper ---
def create_sequences(data, window_size):
    X = []
    if len(data) >= window_size:
        X.append(data[len(data)-window_size:])
    return np.array(X)

# --- Model Path and Globals ---
MODEL_PATH = 'lstm_volatility_model.pth'

# Global scaler and model (loaded once)
scaler = StandardScaler()
lstm_model = None

# --- Load LSTM Model (Once) ---
def load_lstm_model_once():
    global lstm_model
    if lstm_model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"LSTM model file not found at {MODEL_PATH}. Please ensure it's in the correct location.")
        
        # Determine if CUDA is available and set device accordingly
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model_instance = LSTMModel(input_size=1, hidden_size=50, num_layers=2, dropout_prob=0.25)
        
        try:
            state_dict = torch.load(MODEL_PATH, map_location=device)
            model_instance.load_state_dict(state_dict)
        except Exception as e:
            print(f"Standard model loading failed: {e}. Attempting alternative loading.")
            try:
                model_instance = torch.load(MODEL_PATH, map_location=device)
                if not isinstance(model_instance, LSTMModel):
                    model_instance = LSTMModel(input_size=1, hidden_size=50, num_layers=2, dropout_prob=0.25)
                    model_instance.load_state_dict(torch.load(MODEL_PATH, map_location=device))

            except Exception as e_alt:
                 raise RuntimeError(f"Could not load LSTM model from {MODEL_PATH}. Error: {e_alt}")

        model_instance.to(device)
        model_instance.eval()
        lstm_model = model_instance
    return lstm_model, scaler

# --- LSTM Volatility Forecasting ---
def get_lstm_volatility(pred_input: pd.Series, scale_factor: float = 1.0) -> float:
    """
    Returns the next-step forecasted annualized volatility using the trained LSTM.
    pred_input: pandas Series of log returns (e.g., from past 1 year).
    scale_factor: use sqrt(252) for daily → annual conversion.
    """
    global scaler

    if pred_input.empty or len(pred_input) < 21: # Need at least window_size + 1 for one sequence
        print("LSTM Warning: Not enough log returns to compute realized volatility and form a sequence. Returning default.")
        return 0.20 * scale_factor # Fallback, or could be pred_input.std() * scale_factor if enough data for that

    # Calculate Realized Volatility
    volatility_window = 5
    realized_volatility = pred_input.rolling(window=volatility_window).std() * scale_factor 
    realized_volatility.dropna(inplace=True)

    if realized_volatility.empty or len(realized_volatility) < 20: # Need at least window_size for one sequence
        print("LSTM Warning: Not enough realized volatility data points after processing. Returning historical.")
        return pred_input.std() * scale_factor if not pred_input.empty else 0.20 * scale_factor

    # Scale the Realized Volatility
    # IMPORTANT: In a production system, the scaler should be fit on the training data and saved.
    # Here, we re-fit on the current input's realized volatility as a simplification.
    scaled_volatility = scaler.fit_transform(realized_volatility.values.reshape(-1, 1)).flatten()

    # Create Sequences
    window_size = 20 
    X_sequence = create_sequences(scaled_volatility, window_size)

    if X_sequence.shape[0] == 0:
        print("LSTM Warning: Could not create a sequence from the provided data. Returning historical.")
        
        return pred_input.std() * scale_factor if not pred_input.empty else 0.20 * scale_factor

    # Load Model and Make Prediction
    model, _ = load_lstm_model_once() 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X_sequence, dtype=torch.float32).unsqueeze(-1).to(device) 
    with torch.no_grad():
        prediction_scaled = model(X_tensor)
        if isinstance(prediction_scaled, torch.Tensor): 
            prediction_scaled = prediction_scaled.cpu().numpy() 
            if prediction_scaled.ndim == 0: 
                 prediction_scaled_val = prediction_scaled.item()
            elif prediction_scaled.ndim == 1 and len(prediction_scaled) == 1: 
                 prediction_scaled_val = prediction_scaled[0]
            else: 
                 print(f"LSTM Warning: Unexpected prediction shape: {prediction_scaled.shape}. Using first element.")
                 prediction_scaled_val = prediction_scaled[0] if len(prediction_scaled) > 0 else 0.0
        else: 
            prediction_scaled_val = prediction_scaled if np.isscalar(prediction_scaled) else prediction_scaled[0]


    # Inverse Transform the Prediction
    predicted_volatility_actual = scaler.inverse_transform(np.array([[prediction_scaled_val]]))[0,0]
    
    predicted_volatility_actual = max(0.001, predicted_volatility_actual) 

    return float(predicted_volatility_actual)

# --- Testing ---
if __name__ == '__main__':

    dummy_log_returns = pd.Series(np.random.normal(0, 0.01, 200))
    try:
        forecasted_vol = get_lstm_volatility(dummy_log_returns, scale_factor=np.sqrt(252))
        print(f"Forecasted Annualized Volatility: {forecasted_vol:.4f}")
    except Exception as e:
        print(f"Error during LSTM test: {e}")
        
    short_log_returns = pd.Series(np.random.normal(0, 0.01, 10))
    try:
        forecasted_vol_short = get_lstm_volatility(short_log_returns, scale_factor=np.sqrt(252))
        print(f"Forecasted Annualized Volatility (short data): {forecasted_vol_short:.4f}")
    except Exception as e:
        print(f"Error during LSTM short data test: {e}")
