import numpy as np

def simulate_gbm_paths(S0, T, r, sigma, steps, num_paths):
    dt = T / steps
    price_paths = np.zeros((steps + 1, num_paths))
    price_paths[0] = S0
    time_array = np.linspace(0, T, steps + 1)

    for t_idx in range(1, steps + 1):
        Z = np.random.standard_normal(num_paths)
        price_paths[t_idx] = price_paths[t_idx-1] * np.exp((r - 0.5 * sigma**2)*dt + sigma * np.sqrt(dt) * Z)

    return time_array, price_paths
# NEW STUFF:
def calculate_payoffs(terminal_prices, K, option_type):
    if option_type.lower() == 'call':
        payoffs = np.maximum(terminal_prices - K, 0)
    elif option_type.lower() == 'put':
        payoffs = np.maximum(K - terminal_prices, 0)
    else:
        raise ValueError("Invalid option type. Must be 'call' or 'put'.")
    return payoffs

def monte_carlo_option_price(S0, K, T, r, sigma, option_type, steps=100, num_paths=1000):
    _, price_paths = simulate_gbm_paths(S0, T, r, sigma, steps, num_paths)
    terminal_prices = price_paths[-1, :]
    payoffs = calculate_payoffs(terminal_prices, K, option_type)
    discounted_payoffs = np.exp(-r * T) * payoffs
    mc_price = np.mean(discounted_payoffs)
    return mc_price, terminal_prices, payoffs

def get_gbm_analytics(terminal_prices, payoffs, K, T, r, option_type, option_cost):
    """
    Calculates P&L, summary statistics from GBM simulation results.
    option_cost: The price at which the option was assumed to be bought (e.g., from BS or PDE model).
    """
    pnl_values = payoffs - option_cost  # P&L if option bought at option_cost and held to expiry

    # Summary Statistics
    expected_payoff = np.mean(payoffs)
    expected_pnl = np.mean(pnl_values)
    probability_of_profit = np.mean(pnl_values > 0)
    value_at_risk_5_percent = np.percentile(pnl_values, 5) # 5th percentile of P&L
    
    avg_final_stock_price = np.mean(terminal_prices)
    median_final_stock_price = np.median(terminal_prices)

    if option_type.lower() == 'call':
        prob_itm = np.mean(terminal_prices > K)
    else: # put
        prob_itm = np.mean(terminal_prices < K)

    stats = {
        "monte_carlo_price_from_sim": f"${(np.exp(-r * T) * expected_payoff):.4f}", # For consistency check
        "expected_payoff": f"${expected_payoff:.4f}",
        "expected_pnl_vs_model_price": f"${expected_pnl:.4f}",
        "probability_of_profit_vs_model_price": f"{probability_of_profit:.2%}",
        "value_at_risk_5_percent_vs_model_price": f"${value_at_risk_5_percent:.4f}",
        "avg_final_stock_price": f"${avg_final_stock_price:.2f}",
        "median_final_stock_price": f"${median_final_stock_price:.2f}",
        "probability_option_expires_itm": f"{prob_itm:.2%}"
    }
    
    # P&L Histogram Data (frequencies and bin edges)
    # np.histogram returns (counts, bin_edges)
    pnl_counts, pnl_bin_edges = np.histogram(pnl_values, bins=20)
    
    return stats, pnl_counts.tolist(), pnl_bin_edges.tolist()
