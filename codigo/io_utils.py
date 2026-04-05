#io_utils.py

from scipy.io import loadmat, savemat
import re
import scipy.io as sio
from scipy.optimize import linear_sum_assignment
import numpy as np
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