import torch
import torch.nn as nn
from typing import Tuple, Optional, List
import numpy as np

class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        output_size: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False
    ):
        """
        Initialize LSTM model
        
        Parameters:
        -----------
        input_size : int
            Number of input features
        hidden_size : int
            Number of features in the hidden state
        num_layers : int
            Number of recurrent layers
        output_size : int
            Number of output features
        dropout : float
            Dropout probability (0 to 1)
        bidirectional : bool
            Whether to use bidirectional LSTM
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate the size of the last hidden state
        last_hidden_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Fully connected layer
        self.fc = nn.Linear(last_hidden_size, output_size)
        
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_size)
        hidden : Optional[Tuple[torch.Tensor, torch.Tensor]]
            Initial hidden state and cell state
            
        Returns:
        --------
        Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            Output tensor and final hidden state
        """
        # Initialize hidden state if not provided
        if hidden is None:
            batch_size = x.size(0)
            num_directions = 2 if self.bidirectional else 1
            h0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size).to(x.device)
            hidden = (h0, c0)
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Get the last output
        if self.bidirectional:
            # Concatenate the last hidden state from both directions
            last_hidden = torch.cat((lstm_out[:, -1, :self.hidden_size],
                                   lstm_out[:, -1, self.hidden_size:]), dim=1)
        else:
            last_hidden = lstm_out[:, -1, :]
        
        # Apply fully connected layer
        out = self.fc(last_hidden)
        
        return out, hidden

class LSTMTrainer:
    def __init__(
        self,
        model: LSTMModel,
        learning_rate: float = 0.001,
        criterion: nn.Module = nn.MSELoss()
    ):
        """
        Initialize LSTM trainer
        
        Parameters:
        -----------
        model : LSTMModel
            LSTM model instance
        learning_rate : float
            Learning rate for optimizer
        criterion : nn.Module
            Loss function
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> float:
        """
        Perform one training step
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor
        y : torch.Tensor
            Target tensor
            
        Returns:
        --------
        float
            Loss value
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        output, _ = self.model(x)
        loss = self.criterion(output, y)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def predict(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Make predictions
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor
            
        Returns:
        --------
        torch.Tensor
            Predictions
        """
        self.model.eval()
        with torch.no_grad():
            output, _ = self.model(x)
        return output

def prepare_sequences(
    data: np.ndarray,
    seq_length: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare sequences for LSTM training
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    seq_length : int
        Length of input sequences
        
    Returns:
    --------
    Tuple[torch.Tensor, torch.Tensor]
        Input sequences and targets
    """
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length])
    
    return (torch.FloatTensor(sequences),
            torch.FloatTensor(targets))
