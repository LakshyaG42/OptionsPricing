import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import os

# Define the LSTM Model Class (as in the notebook)
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

# Function to create sequences
def create_sequences(data, window_size):
    X = []
    if len(data) >= window_size:
        X.append(data[len(data)-window_size:])
    return np.array(X)

# Path to the trained model
MODEL_PATH = 'lstm_volatility_model.pth' # Assumes the model is in the same directory as app.py or this helper

# Global scaler and model (loaded once)
scaler = StandardScaler()
lstm_model = None

def load_lstm_model_once():
    global lstm_model
    if lstm_model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"LSTM model file not found at {MODEL_PATH}. Please ensure it's in the correct location.")
        
        # Determine if CUDA is available and set device accordingly
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Instantiate the model and move it to the determined device
        model_instance = LSTMModel(input_size=1, hidden_size=50, num_layers=2, dropout_prob=0.25)
        
        # Load the state dictionary. If saved on GPU and loading on CPU, map location.
        # If saved on CPU and loading on GPU, no special mapping needed for model.to(device).
        try:
            state_dict = torch.load(MODEL_PATH, map_location=device)
            model_instance.load_state_dict(state_dict)
        except Exception as e:
            # Fallback for older PyTorch versions or different saving methods if needed
            print(f"Standard model loading failed: {e}. Attempting alternative loading.")
            try:
                # This handles cases where the model might have been saved directly, not just state_dict
                model_instance = torch.load(MODEL_PATH, map_location=device)
                if not isinstance(model_instance, LSTMModel): # Check if loaded object is the model itself
                     # If torch.load loaded a dict, it's likely a state_dict
                    model_instance = LSTMModel(input_size=1, hidden_size=50, num_layers=2, dropout_prob=0.25)
                    model_instance.load_state_dict(torch.load(MODEL_PATH, map_location=device))

            except Exception as e_alt:
                 raise RuntimeError(f"Could not load LSTM model from {MODEL_PATH}. Error: {e_alt}")

        model_instance.to(device) # Ensure model is on the correct device
        model_instance.eval()
        lstm_model = model_instance
    return lstm_model, scaler # Return scaler too, though it's fit per-request below

def get_lstm_volatility(pred_input: pd.Series, scale_factor: float = 1.0) -> float:
    """
    Returns the next-step forecasted annualized volatility using the trained LSTM.
    pred_input: pandas Series of log returns (e.g., from past 1 year).
    scale_factor: use sqrt(252) for daily → annual conversion.
    """
    global scaler # Use the global scaler instance

    if pred_input.empty or len(pred_input) < 21: # Need at least window_size + 1 for one sequence
        print("LSTM Warning: Not enough log returns to compute realized volatility and form a sequence. Returning default.")
        return 0.20 * scale_factor # Fallback, or could be pred_input.std() * scale_factor if enough data for that

    # 1. Calculate Realized Volatility
    volatility_window = 5
    realized_volatility = pred_input.rolling(window=volatility_window).std() * scale_factor # Already annualized by scale_factor
    realized_volatility.dropna(inplace=True)

    if realized_volatility.empty or len(realized_volatility) < 20: # Need at least window_size for one sequence
        print("LSTM Warning: Not enough realized volatility data points after processing. Returning historical.")
        return pred_input.std() * scale_factor if not pred_input.empty else 0.20 * scale_factor

    # 2. Scale the Realized Volatility
    # IMPORTANT: In a production system, the scaler should be fit on the training data and saved.
    # Here, we re-fit on the current input's realized volatility as a simplification.
    scaled_volatility = scaler.fit_transform(realized_volatility.values.reshape(-1, 1)).flatten()

    # 3. Create Sequences
    window_size = 20 
    X_sequence = create_sequences(scaled_volatility, window_size)

    if X_sequence.shape[0] == 0:
        print("LSTM Warning: Could not create a sequence from the provided data. Returning historical.")
        # Fallback to historical volatility of the original log returns
        return pred_input.std() * scale_factor if not pred_input.empty else 0.20 * scale_factor

    # 4. Load Model (if not already loaded) and Make Prediction
    model, _ = load_lstm_model_once() # Scaler is re-fit, so we don't use the one from here
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X_sequence, dtype=torch.float32).unsqueeze(-1).to(device) # Add feature dim and move to device

    with torch.no_grad():
        prediction_scaled = model(X_tensor)
        if isinstance(prediction_scaled, torch.Tensor): # Ensure it's a tensor before calling .item() or .cpu().numpy()
            prediction_scaled = prediction_scaled.cpu().numpy() # Move to CPU if on GPU, then to numpy
            if prediction_scaled.ndim == 0: # If it's a 0-dim array (scalar)
                 prediction_scaled_val = prediction_scaled.item()
            elif prediction_scaled.ndim == 1 and len(prediction_scaled) == 1: # If it's a 1-element array
                 prediction_scaled_val = prediction_scaled[0]
            else: # If it's an array with more elements, take the first one (or handle as error)
                 print(f"LSTM Warning: Unexpected prediction shape: {prediction_scaled.shape}. Using first element.")
                 prediction_scaled_val = prediction_scaled[0] if len(prediction_scaled) > 0 else 0.0
        else: # If it's already a numpy scalar or array from cpu model
            prediction_scaled_val = prediction_scaled if np.isscalar(prediction_scaled) else prediction_scaled[0]


    # 5. Inverse Transform the Prediction
    # Reshape for scaler.inverse_transform which expects 2D array
    predicted_volatility_actual = scaler.inverse_transform(np.array([[prediction_scaled_val]]))[0,0]
    
    # Ensure volatility is positive and reasonable
    predicted_volatility_actual = max(0.001, predicted_volatility_actual) 

    return float(predicted_volatility_actual)

# Example usage (for testing this file directly):
if __name__ == '__main__':
    # Create dummy log return data
    dummy_log_returns = pd.Series(np.random.normal(0, 0.01, 200))
    try:
        forecasted_vol = get_lstm_volatility(dummy_log_returns, scale_factor=np.sqrt(252))
        print(f"Forecasted Annualized Volatility: {forecasted_vol:.4f}")
    except Exception as e:
        print(f"Error during LSTM test: {e}")

    # Test with insufficient data
    short_log_returns = pd.Series(np.random.normal(0, 0.01, 10))
    try:
        forecasted_vol_short = get_lstm_volatility(short_log_returns, scale_factor=np.sqrt(252))
        print(f"Forecasted Annualized Volatility (short data): {forecasted_vol_short:.4f}")
    except Exception as e:
        print(f"Error during LSTM short data test: {e}")
