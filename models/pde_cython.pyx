# filepath: c:\Users\laksh\OneDrive\Documents\GitHub\OptionsPricing\models\pde_cython.pyx
import numpy as np
from scipy.linalg import solve_banded
import math

# It's good practice to cimport numpy
cimport numpy as np

# Cython needs to know the data types of numpy arrays at compile time
DTYPE = np.float
ctypedef np.float_t DTYPE_t

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def bs_put_price(S, K, sigma, T):
    if S <= 0:
        return K
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * norm_cdf(-d2) - S * norm_cdf(-d1)

def solve_tridiagonal_scipy(a, b, c, d):
    n = len(b)
    ab = np.zeros((3, n))
    ab[0, 1:] = c[:n-1] 
    ab[1, :] = b         
    ab[2, :-1] = a[1:]  
    x = solve_banded((1, 1), ab, d)
    return x

cpdef crank_nicolson_call_cython(double S, double K, double sigma, double T, double r, int x_max=200, int N_t=1000):
    cdef double dt = T / N_t
    cdef double S_max = 2 * K + 50
    cdef int Nx = int(x_max / (2 * dt))
    cdef double dx = S_max / Nx
    
    cdef np.ndarray[DTYPE_t, ndim=1] x = np.linspace(0, S_max, Nx + 1)
    cdef np.ndarray[DTYPE_t, ndim=1] V = np.maximum(x - K, 0)
    
    cdef int n = Nx - 1
    cdef np.ndarray[DTYPE_t, ndim=1] alpha = (sigma**2 * x[1:Nx]**2 * dt) / (4 * dx**2)
    
    cdef np.ndarray[DTYPE_t, ndim=1] a = np.zeros(n)
    cdef np.ndarray[DTYPE_t, ndim=1] b = np.ones(n) + 2 * alpha
    cdef np.ndarray[DTYPE_t, ndim=1] c = np.zeros(n)
    
    a[1:] = -alpha[1:]
    c[:-1] = -alpha[:-1]

    cdef np.ndarray[DTYPE_t, ndim=1] V_old = np.copy(V)
    cdef np.ndarray[DTYPE_t, ndim=1] d = np.zeros(n)
    
    cdef int j, step
    cdef double tau

    for step in range(N_t):
        tau = (step + 1) * dt
        V_old[:] = V
        
        d = alpha * V_old[:-2] + (1 - 2 * alpha) * V_old[1:-1] + alpha * V_old[2:]
        
        # Boundary contribution
        d[-1] += alpha[-1] * (S_max - K * math.exp(-r * tau))

        # Solve tridiagonal system
        ab = np.zeros((3, n))
        ab[0, 1:] = c[:-1]
        ab[1, :] = b
        ab[2, :-1] = a[1:]
        V[1:Nx] = solve_banded((1, 1), ab, d)

        V[0] = 0
        V[-1] = S_max - K * math.exp(-r * tau)

    # Interpolate final price at S
    idx = (np.abs(x - S)).argmin()
    return x.tolist(), V.tolist(), V[idx]

cpdef crank_nicolson_put_cython(double S, double K, double sigma, double T, double r, int x_max=3, int N_t=1000):
    cdef double dt = T / N_t
    cdef double S_max = 2 * K + 50
    cdef int Nx = int(x_max / (2 * dt))
    cdef double dx = S_max / Nx
    
    cdef np.ndarray[DTYPE_t, ndim=1] x = np.linspace(0, S_max, Nx + 1)
    cdef np.ndarray[DTYPE_t, ndim=1] V = np.maximum(K - x, 0)
    
    cdef int n = Nx - 1
    cdef np.ndarray[DTYPE_t, ndim=1] alpha = (sigma**2 * x[1:Nx]**2 * dt) / (4 * dx**2)
    
    cdef np.ndarray[DTYPE_t, ndim=1] a = np.zeros(n)
    cdef np.ndarray[DTYPE_t, ndim=1] b = np.ones(n) + 2 * alpha
    cdef np.ndarray[DTYPE_t, ndim=1] c = np.zeros(n)
    
    a[1:] = -alpha[1:]
    c[:-1] = -alpha[:-1]

    cdef np.ndarray[DTYPE_t, ndim=1] V_old = np.copy(V)
    cdef np.ndarray[DTYPE_t, ndim=1] d = np.zeros(n)
    
    cdef int j, step
    cdef double tau

    for step in range(N_t):
        tau = (step + 1) * dt
        V_old[:] = V
        
        d = alpha * V_old[:-2] + (1 - 2 * alpha) * V_old[1:-1] + alpha * V_old[2:]
        
        # Boundary contribution
        d[0] += alpha[0] * K * math.exp(-r * tau)

        # Solve tridiagonal system
        ab = np.zeros((3, n))
        ab[0, 1:] = c[:-1]
        ab[1, :] = b
        ab[2, :-1] = a[1:]
        V[1:Nx] = solve_banded((1, 1), ab, d)

        V[0] = K * math.exp(-r * tau)
        V[-1] = 0

    # Interpolate final price at S
    idx = (np.abs(x - S)).argmin()
    return x.tolist(), V.tolist(), V[idx]
