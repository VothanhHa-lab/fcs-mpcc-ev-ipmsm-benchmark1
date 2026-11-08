"""Test scenarios. All profiles are sampled at the control period Ts.

Bench scenarios (S1, S4, training grid): the speed is imposed by a dynamometer
(mech_free=False), so only the current loop is exercised, as in the manuscript.
Vehicle scenarios (S2, S3, robustness, Monte Carlo): free mechanical dynamics with
J = rotor + vehicle inertia reflected through the gear, shared speed PI + MTPA.
"""
import numpy as np
from .params import vehicle_inertia, road_load_torque, mtpa_id

RPM = 2 * np.pi / 60.0


def _n(T, Ts):
    return int(round(T / Ts))


def step_current(cfg, Ts, rpm=1000.0, iq0=200.0, iq1=600.0, t_step=0.025, T=0.06):
    N = _n(T, Ts)
    t = np.arange(N) * Ts
    iq = np.where(t < t_step, iq0, iq1).astype(np.float64)
    idr = mtpa_id(cfg, iq).astype(np.float64)
    w = np.full(N, rpm * RPM)
    return dict(name="S1_step", iq_ref=iq, id_ref=idr, w_ref=w, TL=np.zeros(N),
                speed_loop=False, mech_free=False, w0=rpm * RPM, iq0=iq0,
                J=cfg["motor"]["J_rotor"], use_mtpa=False, t_event=t_step, T=T)


def steady(cfg, Ts, rpm, torque, T=0.03):
    """Bench steady state at (rpm, torque) on the MTPA / field-weakening trajectory.
    Returns None if the point is outside the torque-speed envelope."""
    from .params import op_point
    op = op_point(cfg, rpm, torque)
    if op is None:
        return None
    idd, iq = float(op[0]), float(op[1])
    N = _n(T, Ts)
    return dict(name=f"steady_{rpm:.0f}rpm_{torque:.0f}Nm", iq_ref=np.full(N, iq),
                id_ref=np.full(N, idd), w_ref=np.full(N, rpm * RPM), TL=np.zeros(N),
                speed_loop=False, mech_free=False, w0=rpm * RPM, iq0=iq,
                J=cfg["motor"]["J_rotor"], use_mtpa=False, t_event=0.0, T=T)


def grade_step(cfg, Ts, rpm=1000.0, grade0=0.0, grade1=0.10, t_step=0.05, T=0.6):
    """Vehicle cruising at constant speed meets a road-grade step (load disturbance)."""
    N = _n(T, Ts)
    t = np.arange(N) * Ts
    w = np.full(N, rpm * RPM)
    gr = np.where(t < t_step, grade0, grade1)
    TL = road_load_torque(cfg, w, gr)
    J = cfg["motor"]["J_rotor"] + vehicle_inertia(cfg)
    Kt = 1.5 * cfg["motor"]["p"] * cfg["motor"]["psi_f"]
    return dict(name="S2_grade", iq_ref=np.zeros(N), id_ref=np.zeros(N), w_ref=w, TL=TL,
                speed_loop=True, mech_free=True, w0=rpm * RPM, iq0=TL[0] / Kt,
                J=J, use_mtpa=True, t_event=t_step, T=T)


def speed_step(cfg, Ts, rpm0=500.0, rpm1=1200.0, t_step=0.05, T=1.0):
    N = _n(T, Ts)
    t = np.arange(N) * Ts
    w = np.where(t < t_step, rpm0, rpm1) * RPM
    TL = road_load_torque(cfg, w)
    J = cfg["motor"]["J_rotor"] + vehicle_inertia(cfg)
    Kt = 1.5 * cfg["motor"]["p"] * cfg["motor"]["psi_f"]
    return dict(name="S3_speed", iq_ref=np.zeros(N), id_ref=np.zeros(N), w_ref=w, TL=TL,
                speed_loop=True, mech_free=True, w0=rpm0 * RPM, iq0=TL[0] / Kt,
                J=J, use_mtpa=True, t_event=t_step, T=T)
