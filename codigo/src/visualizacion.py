#visulizacion.py
import matplotlib.pyplot as plt
import numpy as np
from src.metricas import *
from src.sobi import *

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

#PARA PRUEBAS EN CONSTRAINED NOTEBOOK
def plot_metric_vs_param(df, param_name, metric="corr_rms", title=None):
    plt.figure(figsize=(9, 5))
    for method, g in df.groupby("method"):
        g = g.sort_values(param_name)
        plt.plot(g[param_name], g[metric], marker="o", label=method)

    plt.xlabel(param_name)
    plt.ylabel(metric)
    plt.title(title if title is not None else f"{metric} vs {param_name}")
    plt.grid(True)
    plt.legend()
    plt.show()

def show_case_signals(case, n_samples=None):
    if n_samples is None:
        n_samples = len(case["s1"])

    sl = slice(0, n_samples)
    plt.figure(figsize=(12, 8))

    plt.subplot(4, 1, 1)
    plt.plot(case["s1"][sl], label="s1")
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(case["s2_del"][sl], label="s2 retardada")
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(case["c1"][sl], label="c1")
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.plot(case["c2"][sl], label="c2")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
