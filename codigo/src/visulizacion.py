#visulizacion.py
import matplotlib.pyplot as plt
import numpy as np
from metricas import *
from sobi import *

def plot_autocoralacion(s1, s2,lags,print_val=False):
    """
    Plot de autocorrelacion de dos señales
    """
    r1 = autocorr_norm(s1)
    r2 = autocorr_norm(s2)

    lags = np.arange(len(r1))

    plt.figure(figsize=(10,5))
    plt.plot(lags[:100], r1[:100], label='s1')
    plt.plot(lags[:100], r2[:100], label='s2')
    plt.title("Autocorrelación normalizada")
    plt.xlabel("Lag")
    plt.ylabel("Corr")
    plt.legend()
    plt.grid()
    plt.show()
    
    if print_val:
        for lag in [1, 2, 5, 10, 20]:
            r1 = np.corrcoef(s1[:-lag], s1[lag:])[0,1]
            r2 = np.corrcoef(s2[:-lag], s2[lag:])[0,1]
            print(f"Lag {lag}: s1={r1:.3f}, s2={r2:.3f}")

