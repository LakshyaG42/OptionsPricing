import math

def binomial_price(S, K, T, r, sigma, n_steps=100, option_type="call", american=False):
    dt = T / n_steps
    discount = math.exp(-r * dt)

    # Calculate up/down factors and risk-neutral probability
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp(r * dt) - d) / (u - d)

    # Initialize option values at maturity
    prices = [S * (u**j) * (d**(n_steps - j)) for j in range(n_steps + 1)]
    if option_type == "call":
        values = [max(0, price - K) for price in prices]
    else:
        values = [max(0, K - price) for price in prices]

    # Backward induction
    for i in range(n_steps - 1, -1, -1):
        for j in range(i + 1):
            exercise = 0
            S_ij = S * (u**j) * (d**(i - j))
            hold = discount * (p * values[j + 1] + (1 - p) * values[j])
            if american:
                if option_type == "call":
                    exercise = max(0, S_ij - K)
                else:
                    exercise = max(0, K - S_ij)
                values[j] = max(hold, exercise)
            else:
                values[j] = hold

    return values[0]
