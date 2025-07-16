# distutils: language=c
# cython: language_level=3

import numpy as np
cimport numpy as np
from libc.math cimport exp # Use the C math library for speed

# Define the float type for consistency and precision.
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

# -----------------------------------------------------------------------------
# Private C-Functions for Maximum Performance
# -----------------------------------------------------------------------------

# Use a C-style function directive for cdef functions instead of a decorator
cdef void _thomas_algorithm_cython(
    DTYPE_t[:] a, DTYPE_t[:] b, DTYPE_t[:] c, DTYPE_t[:] d, DTYPE_t[:] out_x
) nogil with gil:
    """
    Solves a tridiagonal system Ax = d using the Thomas Algorithm (TDMA).
    This runs entirely in C and is extremely fast.
    """
    cdef int n = d.shape[0]
    cdef int i

    # Forward elimination pass
    for i in range(1, n):
        w = a[i] / b[i - 1]
        b[i] = b[i] - w * c[i - 1]
        d[i] = d[i] - w * d[i - 1]

    # Backward substitution pass
    out_x[n - 1] = d[n - 1] / b[n - 1]
    for i in range(n - 2, -1, -1):
        out_x[i] = (d[i] - c[i] * out_x[i + 1]) / b[i]


# This pure C function now ONLY contains the hot loop. No numpy creation.
cdef void _run_cn_loop(
    int N_t, int N_x, double K, double r, double dt, double S_max,
    DTYPE_t[:] V, DTYPE_t[:] M2_a, DTYPE_t[:] M2_b, DTYPE_t[:] M2_c, 
    DTYPE_t[:] alpha, DTYPE_t[:] beta, DTYPE_t[:] gamma, bytes option_type
) nogil with gil:
    """
    Contains only the performance-critical time-stepping loop.
    Receives all arrays pre-built from the python-aware wrapper.
    """
    cdef int n = N_x - 1
    cdef DTYPE_t[:] d = np.zeros(n, dtype=DTYPE)
    cdef int step, i

    # Main time-stepping loop
    for step in range(N_t):
        # Calculate the right-hand side vector `d` from the explicit part
        for i in range(n):
            d[i] = alpha[i]*V[i] + (1.0 + beta[i])*V[i+1] + gamma[i]*V[i+2]

        # Apply boundary conditions to the `d` vector
        if option_type == b'call':
            d[n-1] += gamma[n-1] * (S_max - K * exp(-r * (step) * dt)) # Boundary for call
        else: # Put
            d[0] += alpha[0] * K * exp(-r * (step + 1) * dt)

        # Solve the tridiagonal system M2 * V_new = d
        _thomas_algorithm_cython(M2_a.copy(), M2_b.copy(), M2_c.copy(), d, V[1:n+1])

        # Enforce boundary conditions on V for the next iteration
        if option_type == b'call':
            V[0] = 0.0
            V[N_x] = S_max - K * exp(-r * (step + 1) * dt)
        else: # Put
            V[0] = K * exp(-r * (step + 1) * dt)
            V[N_x] = 0.0

# -----------------------------------------------------------------------------
# Public-Facing Python Wrappers
# -----------------------------------------------------------------------------

cpdef tuple crank_nicolson(
    double S, double K, double sigma, double T, double r,
    bytes option_type, int S_max_mult=2, int N_t=1000, int N_x=200):
    """
    Prices a European option using the Crank-Nicolson finite difference method.
    This function handles all NumPy array creation and Python-level logic.

    Args:
        option_type (bytes): b'call' or b'put'

    Returns:
        A tuple containing: (Stock Price Grid, Option Value Grid, Price at S).
    """
    # --- All NumPy creation happens here, in the Python-aware function ---
    cdef double S_max = S_max_mult * K
    cdef double dt = T / N_t
    cdef np.ndarray[DTYPE_t, ndim=1] x = np.linspace(0, S_max, N_x + 1, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] V = np.zeros(N_x + 1, dtype=DTYPE)

    # Set initial condition (payoff)
    if option_type == b'call':
        V[:] = np.maximum(x - K, 0.0)
    elif option_type == b'put':
        V[:] = np.maximum(K - x, 0.0)
    else:
        raise ValueError("option_type must be b'call' or b'put'")

    # Setup coefficients for the tridiagonal matrices
    cdef int n = N_x - 1
    cdef np.ndarray[DTYPE_t, ndim=1] i_vec = np.arange(1, N_x, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] alpha = 0.25 * dt * (sigma**2 * i_vec**2 - r * i_vec)
    cdef np.ndarray[DTYPE_t, ndim=1] beta = -0.5 * dt * (sigma**2 * i_vec**2 + r)
    cdef np.ndarray[DTYPE_t, ndim=1] gamma = 0.25 * dt * (sigma**2 * i_vec**2 + r * i_vec)

    # Constant matrix M2 for the implicit part
    cdef np.ndarray[DTYPE_t, ndim=1] M2_a = -alpha
    cdef np.ndarray[DTYPE_t, ndim=1] M2_b = 1.0 - beta
    cdef np.ndarray[DTYPE_t, ndim=1] M2_c = -gamma

    # --- Call the high-performance C function to run the main loop ---
    _run_cn_loop(N_t, N_x, K, r, dt, S_max, V, M2_a, M2_b, M2_c, alpha, beta, gamma, option_type)

    # Interpolate to find the price at the initial stock price S
    price = np.interp(S, x, V)

    return np.asarray(x), np.asarray(V), price