#constrained_ica.py
import numpy as np

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

def constrained_fastICA(
    X,
    ref=None,                
    constrain_row=0,          
    n_components=None,
    fun="logcosh",
    alpha=1.0,
    max_iter=1000,
    tol=1e-6,
    random_state=0,
    whiten_data=True,
    eps=1e-12,
    hard_ref=True,            
    lambda_ref=0.0,           
):
    """
    Symmetric FastICA con restricción sobre una componente concreta.

    Si ref esta dada:
      q = E[x_w * ref]
    y para la fila k = constrain_row se puede:
      - imponer restricción dura: w^T q = 0
      - añadir penalización blanda: -lambda_ref (w^T q) q

    Devuelve:
      S_hat: (N, n_components)
      W_full: matriz de separación en espacio original
    """
    rng = np.random.default_rng(random_state)

    X = np.asarray(X, dtype=float)
    N, M = X.shape

    if n_components is None:
        n_components = M
    n_components = int(n_components)

    if n_components > M:
        raise ValueError("n_components no puede ser mayor que n_features")

    if whiten_data:
        Xw, whitening_mat, dewhitening_mat, mean_ = whiten(X, eps=eps)
    else:
        mean_ = np.mean(X, axis=0, keepdims=True)
        Xw = X - mean_
        whitening_mat = np.eye(M)
        dewhitening_mat = np.eye(M)

    Xw_use = Xw[:, :n_components] if M != n_components else Xw
    p = Xw_use.shape[1]

    #referencia en 
    q = None
    if ref is not None:
        ref = np.asarray(ref).ravel().astype(float)
        if len(ref) != N:
            raise ValueError("ref debe tener la misma longitud que X")
        ref = ref - np.mean(ref)

        #q = E[x_w ref]
        q = (Xw_use * ref[:, None]).mean(axis=0)
        q_norm = np.linalg.norm(q)
        if q_norm > eps:
            q = q / q_norm
        else:
            q = None

    #inicialización (aleatorio)
    W = rng.standard_normal((n_components, p))
    #decorrelacionamos
    W = sym_decorrelation(W, eps=eps)

    for _ in range(max_iter):
        W_old = W.copy()

        U = Xw_use @ W.T                      # (N, p)
        G, Gp = g_fun(U, fun=fun, alpha=alpha)

        #Actualizamos
        W_new = (G.T @ Xw_use) / N - np.diag(Gp.mean(axis=0)) @ W

        #OJO la restriccion solo aplica a una fila
        k = constrain_row
        wk = W_new[k].copy()

        if q is not None:
            #penalizacion blanda
            if lambda_ref > 0:
                wk = wk - lambda_ref * (wk @ q) * q

            #penalizacion dura
            if hard_ref:
                wk = wk - (wk @ q) * q

        W_new[k] = wk

        #decorrelación simetrica
        W_new = sym_decorrelation(W_new, eps=eps)

        #volver a proyectar la componenente restringida
        if q is not None and hard_ref:
            wk = W_new[k].copy()
            wk = wk - (wk @ q) * q
            norm_wk = np.linalg.norm(wk)
            if norm_wk > eps:
                W_new[k] = wk / norm_wk
            W_new = sym_decorrelation(W_new, eps=eps)

        lim = np.max(np.abs(np.abs(np.diag(W_new @ W_old.T)) - 1.0))
        W = W_new

        if lim < tol:
            break

    W_full = W @ whitening_mat[:p, :]
    Xc = X - mean_
    S_hat = Xc @ W_full.T

    return S_hat, W_full