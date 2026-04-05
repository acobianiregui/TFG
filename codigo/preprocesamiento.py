#preprocesamiento.py
import numpy as np

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

def center(X):
    return X - np.mean(X, axis=0, keepdims=True)


def whiten(X, eps=1e-12):
    """
    X: shape (N, m)
    devuelve:
      Xw: datos blanqueados
      whitening_mat
      dewhitening_mat
      mean_
    """
    X = np.asarray(X, dtype=float)
    mean_ = np.mean(X, axis=0, keepdims=True)
    Xc = X - mean_

    C = np.cov(Xc, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(C)

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals + eps))
    D_sqrt = np.diag(np.sqrt(eigvals + eps))

    whitening_mat = eigvecs @ D_inv_sqrt @ eigvecs.T
    dewhitening_mat = eigvecs @ D_sqrt @ eigvecs.T

    Xw = Xc @ whitening_mat.T
    return Xw, whitening_mat, dewhitening_mat, mean_


def delay_signal(x, delay_samples):
    """
    Retrasa señal SIN circularidad.
    Rellena el inicio con ceros.

    x: array 1D
    delay_samples: entero >= 0
    """
    x = np.asarray(x).ravel()

    if delay_samples <= 0:
        return x.copy()

    y = np.zeros_like(x)
    y[delay_samples:] = x[:-delay_samples]
    return y
