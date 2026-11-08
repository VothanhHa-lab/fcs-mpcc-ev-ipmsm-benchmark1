"""Thin Python wrapper around core.simulate."""
import time
import numpy as np
from . import core
from .params import build

_EMPTY2 = np.zeros((1, 1))
_EMPTY1 = np.zeros(1)


def run(ctrl, cfg, sc, nn=None, lut=None, mismatch=None, seed=0, Ts=None, ctrl_overrides=None):
    pp, mp, cp = build(cfg, sc["J"], mismatch=mismatch, Ts=Ts, ctrl_overrides=ctrl_overrides,
                       use_mtpa=sc["use_mtpa"])
    if nn is None:
        W1, b1, W2, b2, xm, xs, lo, hi = _EMPTY2, _EMPTY1, _EMPTY2, _EMPTY1, _EMPTY1, _EMPTY1, 0.0, 1.0
    else:
        W1, b1, W2, b2 = nn["W1"], nn["b1"], nn["W2"], nn["b2"]
        xm, xs, lo, hi = nn["xm"], nn["xs"], nn["lo"], nn["hi"]
    if lut is None:
        lx, ly, lv = np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.zeros((2, 2))
    else:
        lx, ly, lv = lut["x"], lut["y"], lut["v"]
    Tsv = cp[0]
    N = int(round(sc["T"] / Tsv))

    def rs(a):  # resample scenario arrays if Ts differs from the one used to build them
        a = np.asarray(a, dtype=np.float64)
        if a.shape[0] == N:
            return a
        idx = np.minimum((np.arange(N) * a.shape[0] / N).astype(int), a.shape[0] - 1)
        return a[idx].copy()

    t0 = time.perf_counter()
    log, fine, ncomm, E, ok = core.simulate(
        ctrl, pp, mp, cp, W1, b1, W2, b2, xm, xs, float(lo), float(hi), lx, ly, lv,
        rs(sc["iq_ref"]), rs(sc["id_ref"]), rs(sc["w_ref"]), rs(sc["TL"]),
        bool(sc["speed_loop"]), bool(sc["mech_free"]), float(sc["w0"]), float(sc["iq0"]), int(seed))
    wall = time.perf_counter() - t0
    return dict(ctrl=ctrl, name=core.NAMES[ctrl], log=log, fine=fine, ncomm=ncomm, E=E, ok=ok,
                Ts=Tsv, T=sc["T"], wall=wall, pp=pp, mp=mp, cp=cp, scenario=sc["name"],
                seed=seed, mismatch=mismatch or {})
