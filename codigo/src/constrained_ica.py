#constrained_ica.py
import numpy as np
from src.sobi import _sym_decorrelation
from src.preprocesamiento import whiten

def g_fun_1d(u, fun="logcosh", alpha=1.0):
    """
    No linealidad de FastICA para una sola componente
    """
    if fun == "logcosh":
        g = np.tanh(alpha * u)
        gp = alpha * (1.0 - g**2)
        return g, gp
    elif fun == "cube":
        g = u**3
        gp = 3.0 * u**2
        return g, gp
    elif fun == "exp":
        g = u * np.exp(-(u**2) / 2.0)
        gp = (1.0 - u**2) * np.exp(-(u**2) / 2.0)
        return g, gp
    else:
        raise ValueError(f"fun desconocida: {fun}")
    
def g_fun(U, fun="logcosh", alpha=1.0):
    U = np.asarray(U)

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

def delay_reference(ref, tau):
    """
    Delays reference without circular shift.

    tau > 0 means ref_lag[t] = ref[t - tau]
    """
    ref = np.asarray(ref).ravel()

    if tau == 0:
        return ref.copy()

    ref_lag = np.zeros_like(ref)
    ref_lag[tau:] = ref[:-tau]

    return ref_lag
def build_reference_direction(
    Xw,
    ref,
    max_lag=0,
    smooth=True,
    smooth_win=25,
    eps=1e-12,
):
    """
    Builds q(tau) = E[Xw(t) ref(t - tau)]
    and selects the lag that maximizes ||q(tau)||.
    """

    Xw = np.asarray(Xw, dtype=float)
    ref = np.asarray(ref, dtype=float).ravel()

    if len(ref) != Xw.shape[0]:
        raise ValueError("ref must have the same number of samples as Xw")

    ref = ref - np.mean(ref)

    if smooth:
        kernel = np.ones(smooth_win) / smooth_win
        ref = np.convolve(ref, kernel, mode="same")

    best_q = None
    best_lag = 0
    best_score = -np.inf

    for tau in range(0, max_lag + 1):
        ref_lag = delay_reference(ref, tau)

        # Use only valid region, avoiding zero-padded beginning
        if tau > 0:
            X_valid = Xw[tau:, :]
            r_valid = ref_lag[tau:]
        else:
            X_valid = Xw
            r_valid = ref_lag

        r_valid = r_valid - np.mean(r_valid)

        r_norm = np.std(r_valid)
        if r_norm < eps:
            continue

        r_valid = r_valid / (r_norm + eps)

        q = (X_valid * r_valid[:, None]).mean(axis=0)

        score = np.linalg.norm(q)

        if score > best_score:
            best_score = score
            best_lag = tau
            best_q = q

    if best_q is None:
        return None, None, 0.0

    best_q = best_q / (np.linalg.norm(best_q) + eps)

    return best_q, best_lag, best_score
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

def shift_no_circular(x, lag):
    """
    Desplaza sin circularidad.
    lag > 0  => retrasa
    lag < 0  => adelanta
    """
    x = np.asarray(x).ravel()
    y = np.zeros_like(x)

    if lag == 0:
        return x.copy()

    if lag > 0:
        y[lag:] = x[:-lag]
    else:
        lag = -lag
        y[:-lag] = x[lag:]

    return y

def smooth_reference(ref, mode="moving_average", win_samples=25):
    """
    Suaviza la referencia para que actúe más como activación/envolvente
    que como señal binaria abrupta.
    """
    ref = np.asarray(ref).ravel().astype(float)

    if win_samples <= 1:
        return ref

    if mode == "moving_average":
        kernel = np.ones(win_samples, dtype=float) / win_samples
        return np.convolve(ref, kernel, mode="same")

    raise ValueError(f"Modo de suavizado desconocido: {mode}")




def constrained_fastica_dualref_twounits(
    X,
    ref_s1=None,
    ref_s2=None,
    lambda_pos=0.1,
    lambda_neg=0.1,
    max_lag=0,
    fun="logcosh",
    alpha=1.0,
    whiten_data=True,
    max_iter=1000,
    tol=1e-7,
    random_state=0,
    smooth_ref=True,
    smooth_win=25,
    eps=1e-12,
):
    rng = np.random.default_rng(random_state)

    X = np.asarray(X, dtype=float)
    N, M = X.shape

    #Whitening included if needed
    if whiten_data:
        Xw, whitening_mat, dewhitening_mat, mean_ = whiten(X, eps=eps)
    else:
        mean_ = np.mean(X, axis=0, keepdims=True)
        Xw = X - mean_
        whitening_mat = np.eye(M)
        dewhitening_mat = np.eye(M)

    #Build reference directions, see build_reference_direction() for details
    q_s1, lag_s1, score_s1 = (None, None, 0.0)
    q_s2, lag_s2, score_s2 = (None, None, 0.0)

    if ref_s1 is not None:
        q_s1, lag_s1, score_s1 = build_reference_direction(
            Xw,
            ref_s1,
            max_lag=max_lag,
            smooth=smooth_ref,
            smooth_win=smooth_win,
            eps=eps,
        )

    if ref_s2 is not None:
        q_s2, lag_s2, score_s2 = build_reference_direction(
            Xw,
            ref_s2,
            max_lag=max_lag,
            smooth=smooth_ref,
            smooth_win=smooth_win,
            eps=eps,
        )

    #Initialize W with two rows
    W = rng.standard_normal((2, Xw.shape[1]))

    if q_s1 is not None:
        W[0] = q_s1 + 0.05 * rng.standard_normal(Xw.shape[1])
    if q_s2 is not None:
        W[1] = q_s2 + 0.05 * rng.standard_normal(Xw.shape[1])

    #Symmetric decorrelation
    W = _sym_decorrelation(W)

    converged = False

    for n_iter in range(max_iter):
        W_old = W.copy()

        #Standard FastICA fixed point update
        U = Xw @ W.T
        G, Gp = g_fun(U, fun=fun, alpha=alpha)

        W_new = (G.T @ Xw) / N - np.diag(Gp.mean(axis=0)) @ W

        #constraints for component 1. Target s1, avoid s2
        if q_s1 is not None and lambda_pos > 0:
            W_new[0] += 2 * lambda_pos * np.dot(W[0], q_s1) * q_s1

        if q_s2 is not None and lambda_neg > 0:
            W_new[0] -= 2 * lambda_neg * np.dot(W[0], q_s2) * q_s2

        #constraints for component 2. Target s2, avoid s1
        if q_s2 is not None and lambda_pos > 0:
            W_new[1] += 2 * lambda_pos * np.dot(W[1], q_s2) * q_s2

        if q_s1 is not None and lambda_neg > 0:
            W_new[1] -= 2 * lambda_neg * np.dot(W[1], q_s1) * q_s1

        #Decorrelate
        W_new = _sym_decorrelation(W_new)

        #check convergence
        lim = np.max(np.abs(np.abs(np.diag(W_new @ W_old.T)) - 1.0))

        W = W_new

        if lim < tol:
            converged = True #Solution found
            break

    #Once coverged, restore data
    W_full = W @ whitening_mat.T

    Xc = X - mean_
    S_hat = Xc @ W_full.T
    #Useful information if needed
    info = {
        "q_s1": q_s1,
        "q_s2": q_s2,
        "lag_s1": lag_s1,
        "lag_s2": lag_s2,
        "score_s1": score_s1,
        "score_s2": score_s2,
        "n_iter": n_iter + 1,
        "converged": converged,
        "W_whitened": W,
    }

    return S_hat, W_full, info

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
    W = _sym_decorrelation(W)

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
        W_new = _sym_decorrelation(W_new)

        #volver a proyectar la componenente restringida
        if q is not None and hard_ref:
            wk = W_new[k].copy()
            wk = wk - (wk @ q) * q
            norm_wk = np.linalg.norm(wk)
            if norm_wk > eps:
                W_new[k] = wk / norm_wk
            W_new = _sym_decorrelation(W_new)

        lim = np.max(np.abs(np.abs(np.diag(W_new @ W_old.T)) - 1.0))
        W = W_new

        if lim < tol:
            break

    W_full = W @ whitening_mat[:p, :]
    Xc = X - mean_
    S_hat = Xc @ W_full.T

    return S_hat, W_full

def constrained_fastica_dualref_oneunit(
    X,
    ref_target=None,
    ref_bad=None,
    lambda_pos=0.1,
    lambda_neg=0.1,
    max_lag=0,
    fun="logcosh",
    alpha=1.0,
    whiten_data=True,
    max_iter=1000,
    tol=1e-7,
    random_state=0,
    smooth_ref=True,
    smooth_win=25,
    init_mode="target",
    eps=1e-12,
):
    rng = np.random.default_rng(random_state)

    X = np.asarray(X, dtype=float)
    N, M = X.shape

    if whiten_data:
        Xw, whitening_mat, dewhitening_mat, mean_ = whiten(X, eps=eps)
    else:
        mean_ = np.mean(X, axis=0, keepdims=True)
        Xw = X - mean_
        whitening_mat = np.eye(M)
        dewhitening_mat = np.eye(M)

    q_target, lag_target, score_target = (None, None, 0.0)
    q_bad, lag_bad, score_bad = (None, None, 0.0)

    if ref_target is not None:
        q_target, lag_target, score_target = build_reference_direction(
            Xw, ref_target, max_lag=max_lag, smooth=smooth_ref, smooth_win=smooth_win, eps=eps
        )

    if ref_bad is not None:
        q_bad, lag_bad, score_bad = build_reference_direction(
            Xw, ref_bad, max_lag=max_lag, smooth=smooth_ref, smooth_win=smooth_win, eps=eps
        )

    if init_mode == "target" and q_target is not None:
        w = q_target + 0.05 * rng.standard_normal(Xw.shape[1])
    else:
        w = rng.standard_normal(Xw.shape[1])

    w = w / (np.linalg.norm(w) + eps)
    converged = False

    for n_iter in range(max_iter):
        w_old = w.copy()

        u = Xw @ w
        g, gp = g_fun_1d(u, fun=fun, alpha=alpha)

        w_fastica = (Xw.T @ g) / N - gp.mean() * w

        w_new = w_fastica.copy()

        if q_target is not None and lambda_pos > 0:
            w_new = w_new + 2 * lambda_pos * np.dot(w, q_target) * q_target

        if q_bad is not None and lambda_neg > 0:
            w_new = w_new - 2 * lambda_neg * np.dot(w, q_bad) * q_bad

        nrm = np.linalg.norm(w_new)
        if nrm < eps:
            w_new = rng.standard_normal(Xw.shape[1])
            nrm = np.linalg.norm(w_new)

        w = w_new / (nrm + eps)

        if abs(abs(np.dot(w, w_old)) - 1.0) < tol:
            converged = True
            break

    w_full = whitening_mat.T @ w
    w_full = w_full / (np.linalg.norm(w_full) + eps)

    Xc = X - mean_
    y_hat = Xc @ w_full

    info = {
        "q_target": q_target,
        "q_bad": q_bad,
        "lag_target": lag_target,
        "lag_bad": lag_bad,
        "score_target": score_target,
        "score_bad": score_bad,
        "n_iter": n_iter + 1,
        "converged": converged,
        "w_whitened": w,
    }

    return y_hat, w_full, info

def constrained_fastica_oneunit(
    X,
    ref=None,
    fun="logcosh",
    alpha=1.0,
    lambda_ref=0.2,
    max_lag=0,
    whiten_data=True,
    max_iter=1000,
    tol=1e-7,
    random_state=0,
    smooth_ref=True,
    smooth_win=25,
    eta = 0.2,
    init_mode="ref",  
    eps=1e-12,
):
    """
    FastICA guiado por referencia.
    Parametros
    X : array, shape (N, M)
        Mezclas observadas
    ref : array, shape (N,)
        Referencia de activación para la fuente objetivo
    lambda_ref : float
        Fuerza de atracción hacia q
    max_lag : int
        Retardo máximo permitido en muestras
    init_mode : str
        "ref" inicializa w cerca de q si existe
        "random" inicializa aleatoriamente
    Devuelve
    y_hat : (N,)
        Fuente estimada
    w_full : (M,)
        Vector separador en espacio original
    info : dict
        Información auxiliar
    """
    rng = np.random.default_rng(random_state)

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X debe ser 2D con shape (N, M)")

    N, M = X.shape

    if whiten_data:
        Xw, whitening_mat, dewhitening_mat, mean_ = whiten(X, eps=eps)
    else:
        mean_ = np.mean(X, axis=0, keepdims=True)
        Xw = X - mean_
        whitening_mat = np.eye(M)
        dewhitening_mat = np.eye(M)

    p = Xw.shape[1]

    q = None
    best_lag = 0
    q_score = 0.0

    if ref is not None:
        q, best_lag, q_score = build_reference_direction(
            Xw,
            ref,
            max_lag=max_lag,
            smooth=smooth_ref,
            smooth_win=smooth_win,
            eps=eps
        )

    #Inicializar
    if init_mode == "ref" and q is not None:
        w = q + 0.05 * rng.standard_normal(p)
    else:
        w = rng.standard_normal(p)

    w = w / (np.linalg.norm(w) + eps)

    converged = False

    for n_iter in range(max_iter):
        w_old = w.copy()

        u = Xw @ w                    # (N,)
        g, gp = g_fun_1d(u, fun=fun, alpha=alpha)

        #Actualizacion
        w_fastica = (Xw.T @ g) / N - gp.mean() * w

        #Referencia positiva
        if q is not None and lambda_ref > 0:
            w_new = w_fastica + eta * lambda_ref * q
            w_new = w_new / (np.linalg.norm(w_new) + eps)
        else:
            w_new = w_fastica

        # Normalizar
        norm_w = np.linalg.norm(w_new)
        if norm_w < eps:
            w_new = rng.standard_normal(p)
            norm_w = np.linalg.norm(w_new)

        w = w_new / (norm_w + eps)

        #Normalizar pa converger
        if abs(abs(np.dot(w, w_old)) - 1.0) < tol:
            converged = True
            break

    #Volver al espacio original
    w_full = whitening_mat.T @ w
    w_full = w_full / (np.linalg.norm(w_full) + eps)

    Xc = X - mean_
    y_hat = Xc @ w_full

    info = {
        "q": q,
        "best_lag": best_lag,
        "q_score": q_score,
        "n_iter": n_iter + 1,
        "converged": converged,
        "w_whitened": w,
        "mean_": mean_,
        "whitening_mat": whitening_mat,
        "dewhitening_mat": dewhitening_mat,
    }

    return y_hat, w_full, info