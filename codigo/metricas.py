#metricas.py
import numpy as np
from scipy.optimize import linear_sum_assignment

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

def safe_corr(x, y, eps=1e-12):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    x = x - np.mean(x)
    y = y - np.mean(y)

    sx = np.std(x)
    sy = np.std(y)

    if sx < eps or sy < eps:
        return 0.0

    return np.dot(x, y) / (len(x) * sx * sy)
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
import numpy as np

def correlacion_cruzada(x, y, max_lag): #PARA SOBII!!!!!!!!!!!!!
    """
    Correlación cruzada normalizada entre x e y.

    Parametros
    ----------
    x, y : array
        Señales de misma longitud
    max_lag : int
        Retardo maximo

    Deuvelve
    -------
    lags : np.array
        Retardos (en muestras), de -max_lag a +max_lag
    rxy : np.array
        Correlación cruzada normalizada
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if len(x) != len(y):
        raise ValueError("x e y deben tener la misma longitud")

    #centrar
    x = x - np.mean(x)
    y = y - np.mean(y)

    #correlación cruzada completa
    r = np.correlate(x, y, mode="full")

    # normalización tipo Pearson
    r = r / (np.std(x) * np.std(y) * len(x))

    mid = len(r) // 2
    lags = np.arange(-len(x) + 1, len(x))

    idx = slice(mid - max_lag, mid + max_lag + 1)

    return lags[idx], r[idx]

def safe_corr(x, y, eps=1e-12):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if len(x) != len(y):
        raise ValueError("x e y deben tener la misma longitud")

    #quitar media (ya esta del preprocesamiento pero por si acaso)
    x = x - np.mean(x)
    y = y - np.mean(y)

    #desv stndard
    sx = np.std(x)
    sy = np.std(y)

    #evitar indeterminaciones
    if sx < eps or sy < eps:
        return 0.0

    corr = np.dot(x, y) / (len(x) * sx * sy)

    #clip por si acaso
    return np.clip(corr, -1.0, 1.0)
########################################################################################################################
#RMS
def rms(x): 
    x = np.asarray(x).ravel() 
    return np.sqrt(np.mean(x**2))
def rms_ventanas(signal, fs, window_ms=150, step_ms=50):
    """
    Calcula RMS por ventanas de una señal.

    Parameters
    ----------
    signal : array-like
        Señal de entrada
    fs : int
        Frecuencia de muestreo (Hz)
    window_ms : float
        Tamaño de ventana en ms
    step_ms : float
        Paso entre ventanas en ms

    Returns
    -------
    rms : ndarray
        Serie de RMS por ventanas
    """

    signal = np.asarray(signal)
    
    win = int(fs * window_ms / 1000)
    step = int(fs * step_ms / 1000)

    rms = []

    for start in range(0, len(signal) - win + 1, step):
        segment = signal[start:start+win]
        rms.append(np.sqrt(np.mean(segment**2)))

    return np.array(rms)

def moving_rms(x, win_samples, step_samples=None):
    """
    Envolvente RMS por ventanas.
    Devuelve un array 1D con el RMS de cada ventana.
    """
    x = np.asarray(x).ravel()

    if step_samples is None:
        step_samples = win_samples

    if win_samples <= 0 or step_samples <= 0:
        raise ValueError("win_samples y step_samples deben ser enteros positivos")

    vals = []
    for start in range(0, len(x) - win_samples + 1, step_samples):
        seg = x[start:start + win_samples]
        vals.append(rms(seg))

    if len(vals) == 0:
        return np.array([0.0])

    return np.asarray(vals)

def rms_envelope(x, win_samples, step_samples=None):
    """
    Calcula la envolvente RMS de una señal usando ventanas deslizantes.

    Parameters
    ----------
    x : array-like
        Señal 1D
    win_samples : int
        Tamaño de ventana en muestras
    step_samples : int or None
        Paso entre ventanas (default = win_samples, sin solape)

    Returns
    -------
    env : ndarray
        Envolvente RMS
    """
    x = np.asarray(x).ravel()

    if step_samples is None:
        step_samples = win_samples

    if win_samples <= 0 or step_samples <= 0:
        raise ValueError("win_samples y step_samples deben ser positivos")

    env = []

    for start in range(0, len(x) - win_samples + 1, step_samples):
        segment = x[start:start + win_samples]
        env.append(np.sqrt(np.mean(segment**2)))

    if len(env) == 0:
        return np.array([0.0])

    return np.array(env)
def rms_envelope_metrics(x_true, x_est, fs=1000, window_ms=150, step_ms=50):
    """
    Calcula métricas sobre la envolvente RMS:
      - correlación de envolventes RMS
      - MAE relativa de la envolvente RMS
    """
    win = max(1, int(fs * window_ms / 1000))
    step = max(1, int(fs * step_ms / 1000))

    env_true = moving_rms(x_true, win, step)
    env_est = moving_rms(x_est, win, step)

    n = min(len(env_true), len(env_est))
    env_true = env_true[:n]
    env_est = env_est[:n]

    corr_rms = safe_corr(env_true, env_est)
    mae_rms_env = np.mean(np.abs(env_true - env_est))
    mean_env_true = np.mean(np.abs(env_true)) + 1e-12
    rel_mae_rms_env = mae_rms_env / mean_env_true

    return {
        "corr_rms": corr_rms,
        "mae_rms_env": mae_rms_env,
        "rel_mae_rms_env": rel_mae_rms_env,
        "env_true": env_true,
        "env_est": env_est,
    }
#############################################################################################33
def isi_metric(W, A, eps=1e-12):
    G = W @ A
    P = np.abs(G)
    n = P.shape[0]

    row_max = np.max(P, axis=1, keepdims=True) + eps
    Pr = P / row_max
    row_term = np.sum(Pr, axis=1) - 1.0

    col_max = np.max(P, axis=0, keepdims=True) + eps
    Pc = P / col_max
    col_term = np.sum(Pc, axis=0) - 1.0

    isi = (np.sum(row_term) + np.sum(col_term)) / (2.0 * n * (n - 1) + eps)
    return float(isi), G

def snr_db(s_true, s_est, eps=1e-12):
    s_true = np.asarray(s_true).ravel()
    s_est = np.asarray(s_est).ravel()

    # ajustar escala para comparación justa
    alpha = np.dot(s_est, s_true) / (np.dot(s_est, s_est) + eps)
    s_est_scaled = alpha * s_est

    noise = s_true - s_est_scaled
    p_sig = np.mean(s_true**2)
    p_err = np.mean(noise**2)

    return 10 * np.log10((p_sig + eps) / (p_err + eps))
