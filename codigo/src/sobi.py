#sobi.py
import numpy as np

def _sym_decorrelation(W: np.ndarray) -> np.ndarray:
    """Symmetric decorrelation"""
    s, U = np.linalg.eigh(W @ W.T)
    s = np.maximum(s, 1e-12)
    return (U @ np.diag(1.0 / np.sqrt(s)) @ U.T) @ W

def _joint_diag_jacobi(mats, eps=1e-7, max_sweeps=100):
    """
    Approximate Joint Diagonalization by Jacobi/Givens rotations
    mats: list of (n,n) symmetric matrices to be jointly diagonalized
    Returns: B (n,n) such that for all k, B mats[k] B^T is approx diagonal, see documentation for details
    """
    n = mats[0].shape[0]
    B = np.eye(n)

    for _ in range(max_sweeps):
        improved = False

        #One sweep over pairs (p,q)
        for p in range(n - 1):
            for q in range(p + 1, n):
                #Build 2x2 problem 
                g11 = 0.0
                g12 = 0.0

                for A in mats:
                    app = A[p, p]
                    aqq = A[q, q]
                    apq = A[p, q]
                    #Objective quantities
                    g11 += (app - aqq)
                    g12 += 2.0 * apq

                #If already diagonal, skip
                if abs(g12) <= eps:
                    continue

                #Calculate rotation angle
                theta = 0.5 * np.arctan2(g12, g11)
                c = np.cos(theta)
                s = np.sin(theta)

                if abs(s) <= eps:
                    continue

                improved = True

                #Apply rotation to B
                Bp = B[p, :].copy()
                Bq = B[q, :].copy()
                B[p, :] = c * Bp + s * Bq
                B[q, :] = -s * Bp + c * Bq

                #Update of rows/cols p,q
                for k in range(len(mats)):
                    A = mats[k]

                    #Rotate rows p,q
                    Ap = A[p, :].copy()
                    Aq = A[q, :].copy()
                    A[p, :] = c * Ap + s * Aq
                    A[q, :] = -s * Ap + c * Aq

                    #Rotate cols p,q
                    Ap = A[:, p].copy()
                    Aq = A[:, q].copy()
                    A[:, p] = c * Ap + s * Aq
                    A[:, q] = -s * Ap + c * Aq

                    mats[k] = A

        if not improved:
            break

    return B

def offdiag_energy(M):
    return np.sum(M**2) - np.sum(np.diag(M)**2)

def sobi(X, num_delays=50, delays=None, n_sources=None, eps=1e-7, max_sweeps=100):
    """
    Inputs
    X: array (n_samples, n_channels)
    delays: list of delays (in samples), referenced as T in the pseucode
    n_sources: how many sources to estimate (None => n_channels)
    Returns
    S: (n_samples, n_sources) estimated sources
    W: (n_sources, n_channels) separation matrix
    """
    #Initial check
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D: (n_samples, n_channels)")
    n_samples, n_channels = X.shape
    if n_sources is None:
        n_sources = n_channels
    n_sources = int(n_sources)

    #1 center (if not done already)
    Xc = X - X.mean(axis=0, keepdims=True)

    #2 Whitening (if not whitened already)
    R0 = (Xc.T @ Xc) / n_samples  
    d, E = np.linalg.eigh(R0)
    idx = np.argsort(d)[::-1]
    d = d[idx]
    E = E[:, idx]
    E = E[:, :n_sources]
    d = np.maximum(d[:n_sources], 1e-12)
    Wh = np.diag(1.0 / np.sqrt(d)) @ E.T          
    Xw = Xc @ Wh.T                                

    #3 lagged covariance matrices
    if delays is None:
        delays = list(range(1, int(num_delays) + 1))
    mats = []
    for tau in delays:
        if tau <= 0 or tau >= n_samples:
            continue
        R = (Xw[tau:, :].T @ Xw[:-tau, :]) / (n_samples - tau) 
        R = 0.5 * (R + R.T)  #symmetric
        mats.append(R)
    if len(mats) == 0:
        raise ValueError("No valid set of delays T was provided")

    #4 joint diagonalization (jacobi rotations,see function)
    mats_copy = [A.copy() for A in mats] 
    B = _joint_diag_jacobi(mats_copy, eps=eps, max_sweeps=max_sweeps) 
    B = _sym_decorrelation(B)

    #5 separation
    W = B @ Wh
    S = Xc @ W.T

    return S, W

def autocorr_norm(x):
    x = x - np.mean(x)
    r = np.correlate(x, x, mode='full')
    r = r[r.size // 2:]
    return r / (r[0] + 1e-12)
