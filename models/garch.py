import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Optional

class GARCH:
    def __init__(self, p: int = 1, q: int = 1):
        """
        Initialize GARCH(p,q) model
        
        Parameters:
        -----------
        p : int
            Number of ARCH terms
        q : int
            Number of GARCH terms
        """
        self.p = p
        self.q = q
        self.params = None
        self.variances = None
        
    def _compute_variances(self, returns: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Compute conditional variances using GARCH parameters
        
        Parameters:
        -----------
        returns : np.ndarray
            Array of returns
        params : np.ndarray
            Array of parameters [omega, alpha_1, ..., alpha_p, beta_1, ..., beta_q]
            
        Returns:
        --------
        np.ndarray
            Array of conditional variances
        """
        n = len(returns)
        variances = np.zeros(n)
        
        # Unpack parameters
        omega = params[0]
        alphas = params[1:self.p + 1]
        betas = params[self.p + 1:]
        
        # Initialize with unconditional variance
        variances[0] = np.var(returns)
        
        # Compute conditional variances
        for t in range(1, n):
            arch_term = np.sum(alphas * returns[t-self.p:t][::-1]**2)
            garch_term = np.sum(betas * variances[t-self.q:t][::-1])
            variances[t] = omega + arch_term + garch_term
            
        return variances
    
    def _log_likelihood(self, params: np.ndarray, returns: np.ndarray) -> float:
        """
        Compute log-likelihood of GARCH model
        
        Parameters:
        -----------
        params : np.ndarray
            Array of parameters
        returns : np.ndarray
            Array of returns
            
        Returns:
        --------
        float
            Negative log-likelihood
        """
        variances = self._compute_variances(returns, params)
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * variances) + returns**2 / variances)
        return -log_likelihood  # Return negative for minimization
    
    def fit(self, returns: np.ndarray, 
            initial_params: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit GARCH model to returns data
        
        Parameters:
        -----------
        returns : np.ndarray
            Array of returns
        initial_params : Optional[np.ndarray]
            Initial parameter values
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Estimated parameters and conditional variances
        """
        n_params = 1 + self.p + self.q
        
        # Set default initial parameters if not provided
        if initial_params is None:
            initial_params = np.array([0.1] + [0.1] * self.p + [0.8] * self.q)
        
        # Parameter bounds
        bounds = [(1e-6, None)] + [(0, 1)] * (n_params - 1)
        
        # Optimize parameters
        result = minimize(
            self._log_likelihood,
            initial_params,
            args=(returns,),
            method='L-BFGS-B',
            bounds=bounds
        )
        
        self.params = result.x
        self.variances = self._compute_variances(returns, self.params)
        
        return self.params, self.variances
    
    def forecast(self, returns: np.ndarray, horizon: int = 1) -> np.ndarray:
        """
        Forecast future variances
        
        Parameters:
        -----------
        returns : np.ndarray
            Array of returns
        horizon : int
            Forecast horizon
            
        Returns:
        --------
        np.ndarray
            Forecasted variances
        """
        if self.params is None:
            raise ValueError("Model must be fitted before forecasting")
            
        # Get last p returns and q variances
        last_returns = returns[-self.p:]
        last_variances = self.variances[-self.q:]
        
        # Initialize forecast array
        forecast = np.zeros(horizon)
        
        # Compute forecasts
        for h in range(horizon):
            arch_term = np.sum(self.params[1:self.p + 1] * last_returns[::-1]**2)
            garch_term = np.sum(self.params[self.p + 1:] * last_variances[::-1])
            forecast[h] = self.params[0] + arch_term + garch_term
            
            # Update for next step
            last_returns = np.roll(last_returns, -1)
            last_returns[-1] = np.sqrt(forecast[h]) * np.random.randn()
            last_variances = np.roll(last_variances, -1)
            last_variances[-1] = forecast[h]
            
        return forecast
