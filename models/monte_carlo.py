import numpy as np
from scipy.stats import norm
import yfinance as yf
from datetime import datetime, timedelta

class MonteCarloOptionPricer:
    def __init__(self, 
                 spot_price: float,
                 strike_price: float,
                 time_to_maturity: float,
                 risk_free_rate: float,
                 volatility: float,
                 option_type: str = 'call',
                 num_simulations: int = 10000,
                 num_steps: int = 100):
        """
        Initialize the Monte Carlo Option Pricer.
        
        Args:
            spot_price (float): Current price of the underlying asset
            strike_price (float): Strike price of the option
            time_to_maturity (float): Time to expiration in years
            risk_free_rate (float): Risk-free interest rate (annual)
            volatility (float): Volatility of the underlying asset
            option_type (str): 'call' or 'put'
            num_simulations (int): Number of Monte Carlo simulations
            num_steps (int): Number of time steps in the simulation
        """
        self.spot_price = spot_price
        self.strike_price = strike_price
        self.time_to_maturity = time_to_maturity
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.option_type = option_type.lower()
        self.num_simulations = num_simulations
        self.num_steps = num_steps
        
        if self.option_type not in ['call', 'put']:
            raise ValueError("option_type must be either 'call' or 'put'")

    @classmethod
    def from_ticker(cls, 
                   ticker: str,
                   strike_price: float,
                   quote_date: datetime,
                   expiry_date: datetime,
                   option_type: str = 'call',
                   num_simulations: int = 10000,
                   num_steps: int = 100) -> 'MonteCarloOptionPricer':
        """
        Create a MonteCarloOptionPricer instance from historical ticker data.
        
        Args:
            ticker (str): Stock ticker symbol
            strike_price (float): Strike price of the option
            quote_date (datetime): Date of the option quote
            expiry_date (datetime): Option expiration date
            option_type (str): 'call' or 'put'
            num_simulations (int): Number of Monte Carlo simulations
            num_steps (int): Number of time steps in the simulation
            
        Returns:
            MonteCarloOptionPricer: Configured pricer instance
        """
        # Get stock data
        ticker_data = yf.Ticker(ticker)
        hist = ticker_data.history(start=quote_date, end=quote_date + timedelta(days=1))
        if hist.empty:
            hist = ticker_data.history(end=quote_date + timedelta(days=1), period="5d")
        if hist.empty:
            raise ValueError(f"Could not fetch stock price for {ticker} around {quote_date}")
        spot_price = hist['Close'].iloc[-1]

        # Calculate volatility from historical data
        vol_start = quote_date - timedelta(days=365)
        vol_hist = ticker_data.history(start=vol_start, end=quote_date)
        if len(vol_hist) < 2:
            raise ValueError(f"Not enough historical data for {ticker} to calculate volatility")
        returns = np.log(vol_hist['Close'] / vol_hist['Close'].shift(1))
        volatility = returns.std() * np.sqrt(252)

        # Get risk-free rate
        try:
            irx = yf.Ticker("^IRX")
            r_hist = irx.history(end=quote_date + timedelta(days=1), period="5d")
            if not r_hist.empty:
                risk_free_rate = r_hist['Close'].iloc[-1] / 100.0
            else:
                risk_free_rate = 0.05  # Default to 5% if can't fetch rate
        except:
            risk_free_rate = 0.05  # Default to 5% if can't fetch rate

        # Calculate time to maturity
        time_to_maturity = (expiry_date - quote_date).days / 365.0

        return cls(
            spot_price=spot_price,
            strike_price=strike_price,
            time_to_maturity=time_to_maturity,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            option_type=option_type,
            num_simulations=num_simulations,
            num_steps=num_steps
        )

    def _generate_paths(self) -> np.ndarray:
        """
        Generate price paths using geometric Brownian motion.
        
        Returns:
            np.ndarray: Array of simulated price paths
        """
        dt = self.time_to_maturity / self.num_steps
        drift = (self.risk_free_rate - 0.5 * self.volatility ** 2) * dt
        diffusion = self.volatility * np.sqrt(dt)
        
        # Generate random walks
        random_walks = np.random.normal(0, 1, (self.num_simulations, self.num_steps))
        
        # Calculate price paths
        price_paths = np.zeros((self.num_simulations, self.num_steps + 1))
        price_paths[:, 0] = self.spot_price
        
        for t in range(1, self.num_steps + 1):
            price_paths[:, t] = price_paths[:, t-1] * np.exp(drift + diffusion * random_walks[:, t-1])
            
        return price_paths

    def _calculate_payoff(self, final_prices: np.ndarray) -> np.ndarray:
        """
        Calculate the payoff for each simulation path.
        
        Args:
            final_prices (np.ndarray): Final prices for each simulation path
            
        Returns:
            np.ndarray: Payoff for each path
        """
        if self.option_type == 'call':
            payoffs = np.maximum(final_prices - self.strike_price, 0)
        else:  # put
            payoffs = np.maximum(self.strike_price - final_prices, 0)
            
        return payoffs

    def price_european_option(self) -> tuple[float, float]:
        """
        Price a European option using Monte Carlo simulation.
        
        Returns:
            tuple[float, float]: (option_price, standard_error)
        """
        # Generate price paths
        price_paths = self._generate_paths()
        
        # Calculate payoffs at expiration
        final_prices = price_paths[:, -1]
        payoffs = self._calculate_payoff(final_prices)
        
        # Discount payoffs to present value
        discounted_payoffs = payoffs * np.exp(-self.risk_free_rate * self.time_to_maturity)
        
        # Calculate option price and standard error
        option_price = np.mean(discounted_payoffs)
        standard_error = np.std(discounted_payoffs) / np.sqrt(self.num_simulations)
        
        return option_price, standard_error

    def price_american_option(self) -> tuple[float, float]:
        """
        Price an American option using Monte Carlo simulation with Longstaff-Schwartz method.
        
        Returns:
            tuple[float, float]: (option_price, standard_error)
        """
        # Generate price paths
        price_paths = self._generate_paths()
        
        # Initialize value matrix
        value_matrix = np.zeros_like(price_paths)
        
        # Calculate final payoffs
        value_matrix[:, -1] = self._calculate_payoff(price_paths[:, -1])
        
        # Backward induction
        for t in range(self.num_steps - 1, 0, -1):
            # Calculate immediate exercise value
            exercise_value = self._calculate_payoff(price_paths[:, t])
            
            # Calculate continuation value using regression
            discount_factor = np.exp(-self.risk_free_rate * self.time_to_maturity / self.num_steps)
            continuation_value = value_matrix[:, t + 1] * discount_factor
            
            # Use regression to estimate continuation value
            X = np.column_stack((np.ones_like(price_paths[:, t]), 
                               price_paths[:, t],
                               price_paths[:, t] ** 2))
            beta = np.linalg.lstsq(X, continuation_value, rcond=None)[0]
            estimated_continuation = X @ beta
            
            # Exercise if immediate value is greater than continuation value
            value_matrix[:, t] = np.where(exercise_value > estimated_continuation,
                                        exercise_value,
                                        continuation_value)
        
        # Calculate initial option value
        option_price = np.mean(value_matrix[:, 1]) * np.exp(-self.risk_free_rate * self.time_to_maturity / self.num_steps)
        standard_error = np.std(value_matrix[:, 1]) / np.sqrt(self.num_simulations)
        
        return option_price, standard_error
