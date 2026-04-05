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