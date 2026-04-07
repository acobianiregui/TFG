#constrained_ica.py
import numpy as np

def _g_logcosh(y, alpha=1.0):
    gy = np.tanh(alpha * y)
    gpy = alpha * (1.0 - gy**2)
    return gy, gpy
def g_fun(U, fun="logcosh", alpha=1.0):
    if fun == "logcosh":
        G = np.tanh(alpha * U)
        Gp = alpha * (1.0 - G**2)
        return G, Gp
    elif fun == "cube":
        G = U**3
        Gp = 3.0 * U**2
        return G, Gp
    elif fun == "exp":
        G = U * np.exp(-(U**2) / 2.0)
        Gp = (1.0 - U**2) * np.exp(-(U**2) / 2.0)
        return G, Gp
    else:
        raise ValueError(f"fun desconocida: {fun}")
    
def gram_schmidt_rows_keep_first(W, eps=1e-12):
    W = W.copy()
    m = W.shape[0]

    W[0] = W[0] / (np.linalg.norm(W[0]) + eps)

    for i in range(1, m):
        wi = W[i].copy()
        for j in range(i):
            wi = wi - np.dot(wi, W[j]) * W[j]
        ni = np.linalg.norm(wi)
        if ni < eps:
            wi = np.random.randn(*wi.shape)
            for j in range(i):
                wi = wi - np.dot(wi, W[j]) * W[j]
            ni = np.linalg.norm(wi)
        W[i] = wi / (ni + eps)

    return W