## Funciones auxiliares!!!!

import numpy as np
import re
import scipy.io as sio
from scipy.optimize import linear_sum_assignment


#Funciones para procesar datos y archivos
def find_movement_rep(mat_struct, movement, rep=1, data_field="DAQ_DATA"):
    if not isinstance(movement, str) or len(movement.strip()) == 0:
        raise ValueError("movement debe ser un string no vacío, ej: 'hook', 'handopen'...")

    mv = movement.strip().lower()
    rep_str = str(rep)

    keys = [k for k in mat_struct.keys() if not k.startswith("__")]
    if not keys:
        raise KeyError("El .mat no contiene claves útiles (solo metadatos).")

    def norm(s):
        return re.sub(r"[\s\-]+", "", s.lower())

    mv_norm = norm(mv)

    mv_cand = [k for k in keys if mv_norm in norm(k)]
    if not mv_cand:
        msg = f"No encuentro ninguna clave relacionada con '{movement}'. "
        msg += f"Keys disponibles (primeras 50): {keys[:50]}"
        raise KeyError(msg)

    def rep_score(k):
        kl = norm(k)
        score = 0
        if re.search(rf"_{rep_str}\b", k):         # exact "_1"
            score += 5
        if kl.endswith(rep_str):                    # termina en 1
            score += 4
        if re.search(rf"rep{rep_str}\b", kl):       # contiene "rep1"
            score += 4
        if re.search(rf"\b{rep_str}\b", k):         # aparece como token
            score += 2
        return score

    mv_cand_sorted = sorted(mv_cand, key=lambda k: rep_score(k), reverse=True)

    for k in mv_cand_sorted:
        obj = mat_struct[k]

        if isinstance(obj, np.ndarray) and obj.ndim == 2:
            return obj, k

        if isinstance(obj, dict):
            if data_field in obj:
                X = np.array(obj[data_field])
                if X.ndim == 2:
                    return X, f"{k}.{data_field}"

            for kk in obj.keys():
                kkl = kk.lower()
                if ("daq" in kkl and "data" in kkl) or (data_field.lower() in kkl):
                    X = np.array(obj[kk])
                    if isinstance(X, np.ndarray) and X.ndim == 2:
                        return X, f"{k}.{kk}"

            for kk in obj.keys():
                X = np.array(obj[kk])
                if isinstance(X, np.ndarray) and X.ndim == 2:
                    return X, f"{k}.{kk}"

    raise TypeError(
        f"Encontré claves para '{movement}' (ej: {mv_cand_sorted[:5]}), "
        f"pero no pude extraer una matriz EMG 2D (samples x channels)."
    )
def load_mat(path):
    return sio.loadmat(path, simplify_cells=True)
##########################################################################################################################3
#Funciones de preoprocesamiento 
def eliminar_continua(senal):
    """Quita la componente continua de una señal 1D."""
    return senal - np.mean(senal)
def escalar_canales(emg,metodo="zscore",eps=1e-12):
    """
    Normaliza por canal para comparabilidad e ICA.
    """
    if metodo == "zscore":
        mu = np.mean(emg, axis=0, keepdims=True)
        sd = np.std(emg, axis=0, keepdims=True)
        return (emg - mu) / (sd + eps)

    if metodo == "rms":
        rms = np.sqrt(np.mean(emg**2, axis=0, keepdims=True))
        return emg / (rms + eps)

    if metodo == "mad":
        med = np.median(emg, axis=0, keepdims=True)
        mad = np.median(np.abs(emg - med), axis=0, keepdims=True)
        return (emg - med) / (mad + eps)

    raise ValueError(f"Metodo de escala invalido. Probaste: {metodo}")
def eliminar_outiliers(emg, k=8.0):
    """
    Elimina outliers por canal usando mediana y MAD.
    """
    med = np.median(emg, axis=0, keepdims=True)
    mad = np.median(np.abs(emg - med), axis=0, keepdims=True) + 1e-12
    lo = med - k * mad
    hi = med + k * mad
    return np.clip(emg, lo, hi)
##########################################################################################################################
#Metricas
def amari_error(W, A):
    """
    Amari error entre la matriz de separación W y la de mezcla A.
    Invariante a escala, signo y permutación.
    """
    P = W @ A
    P = np.abs(P)

    n = P.shape[0]

    row = np.sum(P, axis=1) - np.max(P, axis=1)
    col = np.sum(P, axis=0) - np.max(P, axis=0)

    return (row.sum() + col.sum()) / (2 * n)

def corr_matrix(S_true, S_est):
    C = np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            C[i,j] = np.corrcoef(S_true[i], S_est[j])[0,1]
    return C
def match_and_fix(S_true, S_est):
    C = corr_matrix(S_true, S_est)
    cost = 1 - np.abs(C)
    rows, cols = linear_sum_assignment(cost)

    S_al = S_est[cols, :].copy()
    matched_corr = C[rows, cols]
    signs = np.sign(matched_corr)
    signs[signs == 0] = 1
    S_al *= signs[:, None]

    C_final = corr_matrix(S_true, S_al)
    return S_al, C_final
##########################################################################################################################3
#Funciones de SOBI
def _sym_decorrelation(W: np.ndarray) -> np.ndarray:
    """Symmetric decorrelation: W <- (W W^T)^(-1/2) W."""
    s, U = np.linalg.eigh(W @ W.T)
    s = np.maximum(s, 1e-12)
    return (U @ np.diag(1.0 / np.sqrt(s)) @ U.T) @ W

def _joint_diag_jacobi(mats, eps=1e-7, max_sweeps=100):
    """
    Approximate Joint Diagonalization (AJD) by Jacobi/Givens rotations.
    mats: list of (n,n) symmetric matrices to be jointly diagonalized.
    Returns: B (n,n) such that for all k, B mats[k] B^T is ~diagonal.
    """
    n = mats[0].shape[0]
    B = np.eye(n)

    for _ in range(max_sweeps):
        improved = False

        # One sweep over all pairs (p,q)
        for p in range(n - 1):
            for q in range(p + 1, n):
                # Build 2x2 problem accumulated over matrices
                g11 = 0.0
                g22 = 0.0
                g12 = 0.0

                for A in mats:
                    app = A[p, p]
                    aqq = A[q, q]
                    apq = A[p, q]
                    # Objective-driven accumulations
                    g11 += (app - aqq)
                    g12 += 2.0 * apq
                    g22 -= (app - aqq)

                # If already (almost) diagonal w.r.t this pair, skip
                if abs(g12) <= eps:
                    continue

                # Compute rotation angle that reduces off-diagonal terms
                theta = 0.5 * np.arctan2(g12, g11)
                c = np.cos(theta)
                s = np.sin(theta)

                if abs(s) <= eps:
                    continue

                improved = True

                # Apply rotation to B (left-multiply on rows p,q)
                Bp = B[p, :].copy()
                Bq = B[q, :].copy()
                B[p, :] = c * Bp + s * Bq
                B[q, :] = -s * Bp + c * Bq

                # Apply similarity transform to all matrices: A <- G A G^T
                # Efficient update of rows/cols p,q
                for k in range(len(mats)):
                    A = mats[k]

                    # Rotate rows p,q
                    Ap = A[p, :].copy()
                    Aq = A[q, :].copy()
                    A[p, :] = c * Ap + s * Aq
                    A[q, :] = -s * Ap + c * Aq

                    # Rotate cols p,q
                    Ap = A[:, p].copy()
                    Aq = A[:, q].copy()
                    A[:, p] = c * Ap + s * Aq
                    A[:, q] = -s * Ap + c * Aq

                    mats[k] = A

        if not improved:
            break

    return B

def sobi(X, num_delays=50, delays=None, n_sources=None, eps=1e-7, max_sweeps=100):
    """
    SOBI (Second-Order Blind Identification)

    X: array (n_samples, n_channels)
    delays: lista de retardos (en muestras). Si None, usa 1..num_delays.
    n_sources: cuántas fuentes estimar (None => n_channels)

    Returns
    -------
    S: (n_samples, n_sources) fuentes estimadas
    W: (n_sources, n_channels) matriz de separación tal que S = Xc @ W.T
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X debe ser 2D: (n_samples, n_channels)")

    n_samples, n_channels = X.shape
    if n_sources is None:
        n_sources = n_channels
    n_sources = int(n_sources)

    #1 Centrado
    Xc = X - X.mean(axis=0, keepdims=True)

    #2 Whitening 
    R0 = (Xc.T @ Xc) / n_samples  # (ch,ch)
    d, E = np.linalg.eigh(R0)
    idx = np.argsort(d)[::-1]
    d = d[idx]
    E = E[:, idx]

    #3 Reducir al numero de fuentes indicado
    E = E[:, :n_sources]
    d = np.maximum(d[:n_sources], 1e-12)

    Wh = np.diag(1.0 / np.sqrt(d)) @ E.T          
    Xw = Xc @ Wh.T                                

    #3 Matrices de covarianza retardada
    if delays is None:
        delays = list(range(1, int(num_delays) + 1))

    mats = []
    for tau in delays:
        if tau <= 0 or tau >= n_samples:
            continue
        R = (Xw[tau:, :].T @ Xw[:-tau, :]) / (n_samples - tau)  # (src,src)
        R = 0.5 * (R + R.T)  # simetrizar
        mats.append(R)

    if len(mats) == 0:
        raise ValueError("No hay retardos válidos. Ajusta num_delays/delays.")

    #4 Diagonalización conjunta 
    mats_copy = [A.copy() for A in mats] #Copia para no modificar los originales
    B = _joint_diag_jacobi(mats_copy, eps=eps, max_sweeps=max_sweeps)  # (src,src)
    B = _sym_decorrelation(B)

    #5 Separación
    W = B @ Wh
    S = Xc @ W.T

    return S, W
