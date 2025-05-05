import math

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def bs_put_price(S, K, sigma, T):
    if S <= 0:
        return K
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * norm_cdf(-d2) - S * norm_cdf(-d1)

def solve_tridiagonal(a, b, c, d):
    n = len(b)
    cp = [0.0]*(n-1)
    dp = [0.0]*n
    cp[0] = c[0]/b[0]
    dp[0] = d[0]/b[0]
    for i in range(1, n-1):
        denom = b[i] - a[i]*cp[i-1]
        cp[i] = c[i]/denom
        dp[i] = (d[i] - a[i]*dp[i-1]) / denom
    denom = b[-1] - a[-1]*cp[-2]
    dp[-1] = (d[-1] - a[-1]*dp[-2]) / denom
    x = [0.0]*n
    x[-1] = dp[-1]
    for i in reversed(range(n-1)):
        x[i] = dp[i] - cp[i]*x[i+1]
    return x

def crank_nicolson_put(S, K, sigma, T, x_max=3.0, N_t=1000):
    dt = T / N_t
    dx = 2 * dt
    Nx = int(x_max / dx)
    dx = x_max / Nx
    S_max = 2 * K + 50
    dx = S_max / Nx
    x = [i * dx for i in range(Nx + 1)]  # stock price grid

    V = [max(K - xi, 0) for xi in x]

    n = Nx - 1
    alpha = [(sigma ** 2 * x[i] ** 2 * dt) / (4 * dx * dx) for i in range(1, Nx)]
    a = [0.0] + [-alpha[i] for i in range(1, n)]
    b = [1 + 2 * alpha[i] for i in range(n)]
    c = [-alpha[i] for i in range(n - 1)] + [0.0]

    for _ in range(N_t):
        V_old = V[:]
        d = [
            alpha[j] * V_old[j]
            + (1 - 2 * alpha[j]) * V_old[j + 1]
            + alpha[j] * V_old[j + 2]
            for j in range(n)
        ]
        d[0] += alpha[0] * K
        V_int = solve_tridiagonal(a, b, c, d)
        for j in range(n):
            V[j + 1] = V_int[j]
        V[0] = K
        V[-1] = 0

    # Interpolate result at S
    idx = min(range(len(x)), key=lambda i: abs(x[i] - S))
    return x, V, V[idx]

def crank_nicolson_call(S, K, sigma, T, x_max=3.0, N_t=1000):
    dt = T / N_t
    dx = 2 * dt
    Nx = int(x_max / dx)
    dx = x_max / Nx
    S_max = 2 * K + 50
    dx = S_max / Nx
    x = [i * dx for i in range(Nx + 1)]  # stock price grid

    V = [max(xi - K, 0) for xi in x]

    n = Nx - 1
    alpha = [(sigma ** 2 * x[i] ** 2 * dt) / (4 * dx * dx) for i in range(1, Nx)]
    a = [0.0] + [-alpha[i] for i in range(1, n)]
    b = [1 + 2 * alpha[i] for i in range(n)]
    c = [-alpha[i] for i in range(n - 1)] + [0.0]

    for _ in range(N_t):
        V_old = V[:]
        d = [
            alpha[j] * V_old[j]
            + (1 - 2 * alpha[j]) * V_old[j + 1]
            + alpha[j] * V_old[j + 2]
            for j in range(n)
        ]
        d[0] += alpha[0] * 0  # left boundary: V[0] = 0
        V_int = solve_tridiagonal(a, b, c, d)
        for j in range(n):
            V[j + 1] = V_int[j]
        V[0] = 0  # left boundary (S = 0)
        V[-1] = x[-1] - K  # right boundary approx

    # Interpolate result at S
    idx = min(range(len(x)), key=lambda i: abs(x[i] - S))
    return x, V, V[idx]
