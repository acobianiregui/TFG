#evaluacion.py
import numpy as np
from sklearn.decomposition import FastICA
from src.metricas import *
import pandas as pd
from src.constrained_ica import *
from src.generador import *
#RUN algortimos

# FastICA
def run_fastica(X, n_components=2, random_state=0):
    ica = FastICA(
        n_components=n_components,
        whiten="unit-variance",
        random_state=random_state,
        max_iter=2000,
        tol=1e-5
    )
    S_hat = ica.fit_transform(X)
    W = ica.components_
    return S_hat, W
def align_estimated_to_true(S_est, S_true):
    """
    Reordena y cambia signo de S_est para maximizar correlación
    con S_true.
    Ambos: shape (N, 2)
    """
    S_est = np.asarray(S_est)
    S_true = np.asarray(S_true)

    perms = [
        [0, 1],
        [1, 0],
    ]

    best_score = -np.inf
    best_S = None

    for perm in perms:
        S_tmp = S_est[:, perm].copy()
        score = 0.0

        for j in range(2):
            c = safe_corr(S_tmp[:, j], S_true[:, j])
            if c < 0:
                S_tmp[:, j] *= -1
                c = -c
            score += c

        if score > best_score:
            best_score = score
            best_S = S_tmp

    return best_S


def align_sources(S_est, S_true):
    """
    Reordena y corrige signo de las fuentes estimadas
    para maximizar correlación con las verdaderas.
    """
    perms = [[0, 1], [1, 0]]

    best = None
    best_score = -1

    for p in perms:
        S = S_est[:, p].copy()
        score = 0

        for i in range(2):
            c = safe_corr(S[:, i], S_true[:, i])

            if c < 0:
                S[:, i] *= -1
                c = -c

            score += c

        if score > best_score:
            best_score = score
            best = S

    return best


def evaluate_method(name, S_hat, S_true, fs=1000, window_ms=150, step_ms=50):
    """
    Alinea SIN reescalar y devuelve métricas:
      - corr temporal
      - RMS global
      - error relativo RMS
      - correlación de la envolvente RMS
    """
    S_eval = align_estimated_to_true(S_hat, S_true)

    rows = []
    labels = ["s1", "s2"]

    env_info = {}

    for j, lab in enumerate(labels):
        x_true = S_true[:, j]
        x_est = S_eval[:, j]

        corr_val = safe_corr(x_est, x_true)
        rms_true = rms(x_true)
        rms_est = rms(x_est)
        rms_rel_err = np.abs(rms_est - rms_true) / (rms_true + 1e-12)

        env_metrics = rms_envelope_metrics(
            x_true, x_est,
            fs=fs,
            window_ms=window_ms,
            step_ms=step_ms
        )

        rows.append({
            "method": name,
            "source": lab,
            "corr": corr_val,
            "corr_rms": env_metrics["corr_rms"],
            "rms_true": rms_true,
            "rms_est": rms_est,
            "rms_rel_error": rms_rel_err,
            "rel_mae_rms_env": env_metrics["rel_mae_rms_env"],
        })

        env_info[lab] = env_metrics

    return pd.DataFrame(rows), S_eval, env_info

def comparar_rms_ventanas(x, y, fs, window_ms=150, step_ms=50):
    """
    Calcula RMS por ventanas de dos señales y devuelve su correlación.

    Parameters
    ----------
    x, y : array-like
        Señales a comparar
    fs : int
        Frecuencia de muestreo
    window_ms : float
        Tamaño de ventana RMS
    step_ms : float
        Paso entre ventanas

    Returns
    -------
    rms_x : ndarray
    rms_y : ndarray
    corr : float
    """

    x = np.asarray(x) - np.mean(x)
    y = np.asarray(y) - np.mean(y)

    rms_x = rms_ventanas(x, fs, window_ms, step_ms)
    rms_y = rms_ventanas(y, fs, window_ms, step_ms)

    min_len = min(len(rms_x), len(rms_y))

    rms_x = rms_x[:min_len]
    rms_y = rms_y[:min_len]

    corr = np.corrcoef(rms_x, rms_y)[0,1]

    return rms_x, rms_y, corr

#PARA CONSTRAINED NOTEBOOK
def run_methods_on_case(case, lambda_ref=5.0, random_state=0):
    """
    Ejecuta:
      - FastICA
      - constrained FastICA con anti-referencia del contaminante
      - constrained FastICA con referencia imperfecta (anti-ref)
    """
    X = case["X"]
    S_true = case["S_true"]

    # Baseline ICA
    S_ica, W_ica = run_fastica(X, random_state=random_state)
    df_ica, S_ica_al, env_ica = evaluate_method("FastICA", S_ica, S_true)

    # Constrained usando anti-referencia del contaminante
    S_con_bad, W_con_bad = constrained_fastICA(
        X,
        ref=case["ref_bad"],
        constrain_row=0,
        n_components=2,
        random_state=random_state,
        hard_ref=True,
        lambda_ref=lambda_ref,
    )
    df_con_bad, S_con_bad_al, env_con_bad = evaluate_method(
        "Constrained (anti-ref s2)", S_con_bad, S_true
    )

    # Constrained usando referencia imperfecta tratada como anti-ref
    S_con_imp, W_con_imp = constrained_fastICA(
        X,
        ref=case["ref_good_imperfect"],
        constrain_row=0,
        n_components=2,
        random_state=random_state,
        hard_ref=True,
        lambda_ref=lambda_ref,
    )
    df_con_imp, S_con_imp_al, env_con_imp = evaluate_method(
        "Constrained (ref imperfecta)", S_con_imp, S_true
    )

    df = pd.concat([df_ica, df_con_bad, df_con_imp], ignore_index=True)

    return {
        "df": df,
        "S_ica": S_ica_al,
        "S_con_bad": S_con_bad_al,
        "S_con_imp": S_con_imp_al,
        "env_ica": env_ica,
        "env_con_bad": env_con_bad,
        "env_con_imp": env_con_imp,
    }
##PARA PRUEBAS DE LOS PARAMETROS
def run_grid(param_name, values, base_case_kwargs, lambda_ref=5.0, random_state=0):
    """
    Barre un parámetro y devuelve un DataFrame con métricas de s1.
    """
    rows = []

    for v in values:
        kwargs = dict(base_case_kwargs)
        kwargs[param_name] = v

        case = build_case(random_state=random_state, **kwargs)
        out = run_methods_on_case(case, lambda_ref=lambda_ref, random_state=random_state)

        s1_df = out["df"][out["df"]["source"] == "s1"].copy()
        s1_df[param_name] = v
        rows.append(s1_df)

    return pd.concat(rows, ignore_index=True)

## Cuando se usa build_case() esta funcion viene bien
def summarize_s1(df):
    """
    Resumen solo para s1, que es la fuente objetivo.
    """
    cols = ["method", "corr", "corr_rms", "rms_rel_error", "rel_mae_rms_env"]
    return (
        df[df["source"] == "s1"][cols]
        .sort_values(["corr_rms", "corr"], ascending=False)
        .reset_index(drop=True)
    )