#generador.py

import signal

import numpy as np
from src.preprocesamiento import *

def condition_signals(u1,u2,pattern=[0, 3, 2,1, 3, 3, 1, 2, 3, 0,2, 3,1],dur_block1=0.5,dur_block2=0.5,fs=1000):
    N = min(len(u1), len(u2))
    u1 = u1[:N]

    u2 = u2[:N]
    N = len(u1)
    t = np.arange(N) / fs

    m1 = np.zeros(N)
    m2 = np.zeros(N)
    m3 = np.zeros(N)
    
    block = int(dur_block1 * fs)
    block2 = int(dur_block2 * fs)
    for i, state in enumerate(pattern):
        a = i * block
        a2= i * block2
        b = min((i + 1) * block, N)
        b2= min((i + 1) * block2, N)
        if state == 1:
            m1[a:b] = 1.0
            m3[a:b] = 1.0
        elif state == 2:
            m2[a2:b2] = 1.0
        elif state == 3:
            m1[a:b] = 1.0
            m3[a:b] = 1.0
            m2[a2:b2] = 1.0


    #Apply binary mask
    s1 = m1 * u1
    s2 = m2 * u2
    return s1,s2
"""
SPANISH VERSION (USED IN TESTING NOTEBOOK)
def construir_senales(u1,u2_raw,patron=[0, 3, 2,1, 3, 3, 1, 2, 3, 0,2, 3,1],
                      dur_bloque1=0.5,dur_bloque2=0.5,fs=1000):
    N = min(len(u1), len(u2_raw))
    u1 = u1[:N]

    u2_raw = u2_raw[:N]
    N = len(u1)
    t = np.arange(N) / fs

    m1 = np.zeros(N)
    m2 = np.zeros(N)
    m3 = np.zeros(N)
    
    bloque = int(dur_bloque1)
    bloque2 = int(dur_bloque2)
    for i, estado in enumerate(patron):
        a = i * bloque
        a2= i * bloque2
        b = min((i + 1) * bloque, N)
        b2= min((i + 1) * bloque2, N)
        if estado == 1:
            m1[a:b] = 1.0
            m3[a:b] = 1.0
        elif estado == 2:
            m2[a2:b2] = 1.0
        elif estado == 3:
            m1[a:b] = 1.0
            m3[a:b] = 1.0
            m2[a2:b2] = 1.0


    #CONSTRUCCION DE SEÑALES
    s1 = m1 * u1
    s2 = m2 * u2_raw
    return s1,s2
"""


def generate_artificial_signals(fs=1000, duration=10, seed=None, eps=1e-12):
    """
    Generate two artificial EMG source signals with different frequency
    bands and envelopes

    Returns
    t : ndarray
        Time vector.
    S_true : ndarray
        Source matrix with shape (N, 2).
    """

    rng = np.random.default_rng(seed)

    t = np.arange(0, duration, 1 / fs)
    N = len(t)

    #s1: fast component
    raw_s1 = rng.normal(0, 1, N)
    b1, a1 = signal.butter(2, [90, 180], btype="bandpass", fs=fs)
    s1 = signal.filtfilt(b1, a1, raw_s1)
    s1 *= 0.5 * (1 + np.sin(2 * np.pi * 0.9 * t))

    #s2: slow component
    raw_s2 = rng.normal(0, 1, N)
    b2, a2 = signal.butter(2, [20, 50], btype="bandpass", fs=fs)
    s2 = signal.filtfilt(b2, a2, raw_s2)
    s2 *= 0.6 * (1 + np.cos(2 * np.pi * 0.15 * t))

    #RMS normalization
    s1 = s1 / (np.sqrt(np.mean(s1**2)) + eps)
    s2 = s2 / (np.sqrt(np.mean(s2**2)) + eps)

    S_true = np.column_stack([s1, s2])

    return t, S_true, s1, s2

def build_case(
    u1, u2_raw, fs=1000, beta=1.0, a11=1.0, a21=0.01,
    tau_ms=0.0, noise_std=0.0, pattern=None, block_ms=200,
    target_scale=1.0, contam_scale=1.0, ref_flip_prob=0.0,
    ref_fn_prob=0.0, ref_fp_prob=0.0, random_state=0
):
    """
    THIS FUNCTION COMBINES ALL METHODOLOGY STEPS.
    Mainly used for constrained ICA testing, but can be used for any algorithm.
    Builds observed channels by mixing original sources
        c1 = a11*s1 + beta*s2(t-tau) + n1
        c2 = a21*s1 + 1.0*s2 + n2

    Returns:
        dictionary with signals, mixtures, masks and references
    """
    rng = np.random.default_rng(random_state)

    u1 = np.asarray(u1).ravel().astype(float)
    u2_raw = np.asarray(u2_raw).ravel().astype(float)
    N = min(len(u1), len(u2_raw))
    u1 = eliminar_continua(u1[:N])
    u2_raw = eliminar_continua(u2_raw[:N])

    if pattern is None:
        pattern = [0, 1, 2,1, 3, 2, 1, 2, 3, 0,2, 3,1]

    block = int(block_ms * fs / 1000)
    m1 = np.zeros(N)
    m2 = np.zeros(N)

    for i, estado in enumerate(pattern):
        a = i * block
        b = min((i + 1) * block, N)
        if a >= N:
            break
        if estado == 1:
            m1[a:b] = 1.0
        elif estado == 2:
            m2[a:b] = 1.0
        elif estado == 3:
            m1[a:b] = 1.0
            m2[a:b] = 1.0

    s1 = target_scale * m1 * u1
    s2 = contam_scale * m2 * u2_raw

    tau = int(fs * tau_ms / 1000)
    s2_del = delay_signal(s2, tau)

    n1 = noise_std * rng.standard_normal(N)
    n2 = noise_std * rng.standard_normal(N)

    c1 = a11 * s1 + beta * s2_del + n1
    c2 = a21 * s1 + 1.0 * s2 + n2

    X = np.column_stack([c1, c2])
    S_true = np.column_stack([s1, s2_del])

    #Ideal binary references
    ref_good = (m1 > 0).astype(float)
    ref_bad = (m2 > 0).astype(float)

    #Imperfections
    ref_imp = ref_good.copy()
    idx = np.arange(N)

    #false negatives
    pos = idx[ref_imp == 1]
    if len(pos) > 0 and ref_fn_prob > 0:
        k = int(ref_fn_prob * len(pos))
        if k > 0:
            off = rng.choice(pos, size=k, replace=False)
            ref_imp[off] = 0.0

    #false positives
    neg = idx[ref_imp == 0]
    if len(neg) > 0 and ref_fp_prob > 0:
        k = int(ref_fp_prob * len(neg))
        if k > 0:
            on = rng.choice(neg, size=k, replace=False)
            ref_imp[on] = 1.0

    #aribitrary flips
    if ref_flip_prob > 0:
        k = int(ref_flip_prob * N)
        if k > 0:
            flip = rng.choice(idx, size=k, replace=False)
            ref_imp[flip] = 1.0 - ref_imp[flip]

    return {
        "u1": u1, "u2_raw": u2_raw, "m1": m1, "m2": m2,
        "s1": s1, "s2": s2, "s2_del": s2_del,
        "X": X, "S_true": S_true,
        "c1": c1, "c2": c2,
        "ref_good": ref_good,
        "ref_bad": ref_bad,
        "ref_good_imperfect": ref_imp,
        "tau": tau
    }