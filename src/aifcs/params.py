"""Configuration -> numeric parameter vectors for core.simulate."""
import json
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(path=None):
    path = path or os.path.join(ROOT, "configs", "default.json")
    with open(path) as f:
        return json.load(f)


def vehicle_inertia(cfg):
    v = cfg["vehicle"]
    return v["m"] * v["Rw"] ** 2 / v["G"] ** 2


def road_load_torque(cfg, w_m, grade=0.0):
    """Road-load torque referred to the motor shaft (N m) for motor speed w_m (rad/s)."""
    v = cfg["vehicle"]
    speed = w_m * v["Rw"] / v["G"]
    th = np.arctan(grade)
    F = (v["m"] * v["g"] * v["Cr"] * np.cos(th) + 0.5 * v["rho"] * v["Cd"] * v["A"] * speed ** 2
         + v["m"] * v["g"] * np.sin(th))
    return F * v["Rw"] / v["G"]


def build(cfg, J_total, mismatch=None, Ts=None, ctrl_overrides=None, use_mtpa=True):
    """Return (pp, mp, cp).

    mismatch: dict of multiplicative factors applied to the PLANT only
              (keys Rs, Ld, Lq, psi, J, Vdc). The controller keeps nominal values.
    """
    m, inv, c = cfg["motor"], cfg["inverter"], cfg["control"]
    mm = {"Rs": 1.0, "Ld": 1.0, "Lq": 1.0, "psi": 1.0, "J": 1.0, "Vdc": 1.0}
    if mismatch:
        mm.update(mismatch)
    Ts = Ts or c["Ts"]
    pp = np.array([m["Rs"] * mm["Rs"], m["Ld"] * mm["Ld"], m["Lq"] * mm["Lq"], m["psi_f"] * mm["psi"],
                   m["p"], J_total * mm["J"], m["B"], inv["Vdc"] * mm["Vdc"], inv["dead_time"],
                   inv["Esw_ref"], inv["Vref"], inv["Iref"], inv["V0"], inv["r_on"]], dtype=np.float64)
    mp = np.array([m["Rs"], m["Ld"], m["Lq"], m["psi_f"], J_total, m["B"]], dtype=np.float64)
    # current-loop bandwidth: configured value, capped at fs/12 so that slow PWM rates stay stable
    wc = 2 * np.pi * min(c["pi_bandwidth_hz"], 1.0 / (12.0 * Ts))
    KpD, KiD = m["Ld"] * wc, m["Rs"] * wc
    KpQ, KiQ = m["Lq"] * wc, m["Rs"] * wc
    Kt = 1.5 * m["p"] * m["psi_f"]
    wb = 2 * np.pi * c["speed_bandwidth_hz"]
    Kpw = J_total * wb / Kt
    Kiw = Kpw * wb / 4.0
    wo = 2 * np.pi * c["observer_bandwidth_hz"]
    l1 = 2 * 0.707 * wo
    l2 = J_total * wo ** 2
    cp = np.array([Ts, c["nsub"], KpD, KiD, KpQ, KiQ, Kpw, Kiw, m["Imax"],
                   1.0, 1.0, 0.0, c["sigma_i"], c["sigma_w"], c["speed_div"], c["log_div"],
                   l1, l2, c["Ilim_hard_factor"] * m["Imax"], 1.0 if use_mtpa else 0.0,
                   iq_cmd_for_torque(cfg, m.get("T_peak", 430.0))],
                  dtype=np.float64)
    if ctrl_overrides:
        idx = {"lam_d": 9, "lam_q": 10, "lam_sw": 11, "sigma_i": 12, "sigma_w": 13, "nsub": 1}
        for k, v in ctrl_overrides.items():
            cp[idx[k]] = v
    return pp, mp, cp


def mtpa_id(cfg, iq):
    m = cfg["motor"]
    dL = m["Lq"] - m["Ld"]
    a = m["psi_f"] / (2 * dL)
    return a - np.sqrt(a * a + np.asarray(iq) ** 2)


def _torque(m, i_d, i_q):
    return 1.5 * m["p"] * (m["psi_f"] * i_q + (m["Ld"] - m["Lq"]) * i_d * i_q)


def op_point(cfg, rpm, torque, Vdc=None):
    """Steady (id, iq) reference for a torque request at a speed, using the same
    MTPA + field-weakening + current-limit generator as the closed loop.
    Returns None when the torque is not reachable at that speed."""
    from .core import ref_mtpa_fw
    m = cfg["motor"]
    Vdc = Vdc or cfg["inverter"]["Vdc"]
    we = m["p"] * rpm * 2 * np.pi / 60
    gen = lambda iqc: ref_mtpa_fw(iqc, we, Vdc, m["Ld"], m["Lq"], m["psi_f"], m["Imax"])
    if torque <= 0:
        return gen(0.0)
    lo, hi = 0.0, 2.0 * m["Imax"]
    if torque > max_torque(cfg, rpm, Vdc) + 1e-9:
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _torque(m, *gen(mid)) < torque:
            lo = mid
        else:
            hi = mid
    return gen(hi)


def iq_cmd_for_torque(cfg, T):
    """Speed-loop output (iq command) that yields torque T on the MTPA curve at standstill."""
    m = cfg["motor"]
    lo, hi = 0.0, m["Imax"]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        dL = m["Lq"] - m["Ld"]
        a = m["psi_f"] / (2 * dL)
        idd = a - np.sqrt(a * a + mid * mid)
        if _torque(m, idd, mid) < T:
            lo = mid
        else:
            hi = mid
    return hi


def max_torque(cfg, rpm, Vdc=None):
    from .core import ref_mtpa_fw
    m = cfg["motor"]
    Vdc = Vdc or cfg["inverter"]["Vdc"]
    we = m["p"] * rpm * 2 * np.pi / 60
    env = _torque(m, *ref_mtpa_fw(2.0 * m["Imax"], we, Vdc, m["Ld"], m["Lq"], m["psi_f"], m["Imax"]))
    wm = max(rpm, 1.0) * 2 * np.pi / 60
    return min(env, m.get("T_peak", 430.0), m.get("P_peak", 211e3) / wm)
