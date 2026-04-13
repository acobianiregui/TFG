#generador.py

import numpy as np
from src.preprocesamiento import *
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
    # ejemplo por bloques de 0.5 
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

def build_case(
    u1, u2_raw, fs=1000, beta=1.0, a11=1.0, a21=0.01,
    tau_ms=0.0, noise_std=0.0, pattern=None, block_ms=200,
    target_scale=1.0, contam_scale=1.0, ref_flip_prob=0.0,
    ref_fn_prob=0.0, ref_fp_prob=0.0, random_state=0
):
    """
    Construye un caso sintético coherente con el modelo:
        c1 = a11*s1 + beta*s2(t-tau) + n1
        c2 = a21*s1 + 1.0*s2 + n2

    Devuelve:
        dict con señales, mezclas, máscaras y referencias
    """
    rng = np.random.default_rng(random_state)

    u1 = np.asarray(u1).ravel().astype(float)
    u2_raw = np.asarray(u2_raw).ravel().astype(float)
    N = min(len(u1), len(u2_raw))
    u1 = eliminar_continua(u1[:N])
    u2_raw = eliminar_continua(u2_raw[:N])

    if pattern is None:
        pattern = [0, 3, 2, 1, 3, 3, 1, 2, 3, 0, 2, 3, 1]

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

    # Referencias binarias ideales
    ref_good = (m1 > 0).astype(float)
    ref_bad = (m2 > 0).astype(float)

    # Imperfecciones en la referencia buena
    ref_imp = ref_good.copy()
    idx = np.arange(N)

    # falsos negativos
    pos = idx[ref_imp == 1]
    if len(pos) > 0 and ref_fn_prob > 0:
        k = int(ref_fn_prob * len(pos))
        if k > 0:
            off = rng.choice(pos, size=k, replace=False)
            ref_imp[off] = 0.0

    # falsos positivos
    neg = idx[ref_imp == 0]
    if len(neg) > 0 and ref_fp_prob > 0:
        k = int(ref_fp_prob * len(neg))
        if k > 0:
            on = rng.choice(neg, size=k, replace=False)
            ref_imp[on] = 1.0

    # flips arbitrarios
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