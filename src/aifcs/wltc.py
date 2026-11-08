"""Quasi-static WLTC Class 3b energy evaluation using SIMULATED drive-loss maps.

Each controller's loss map P_loss(speed, |torque|) = P_cu + P_cond + P_sw comes from
switching-level steady-state simulations (see reproduce.py, section 6). The cycle is
then evaluated second by second (standard quasi-static method):
  F = m a + m g Cr + 0.5 rho Cd A v^2 ;  T_m = F Rw / G + J_rotor * dw/dt
Assumptions (also in README): gear efficiency 1, losses symmetric in motoring and
regeneration, regeneration limited by the motor envelope (rest = friction brake),
inverter disabled at standstill (zero loss), mechanical losses not included. The loss grid passed to energy()
may already include the simulated stator iron loss (reproduce.py section 6a adds P_fe from ironloss.py); rotor/magnet
iron loss is not modelled.
"""
import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .params import ROOT, max_torque

RPM = 2 * np.pi / 60.0


def load_cycle():
    v = np.loadtxt(os.path.join(ROOT, "data", "wltc_class3b.csv"))  # km/h at 1 Hz
    return v / 3.6


def demand(cfg):
    veh, m = cfg["vehicle"], cfg["motor"]
    v = load_cycle()
    a = np.gradient(v)                      # m/s^2 (1 Hz grid)
    F = veh["m"] * a + veh["m"] * veh["g"] * veh["Cr"] * (v > 0.01) + 0.5 * veh["rho"] * veh["Cd"] * veh["A"] * v ** 2
    w = v / veh["Rw"] * veh["G"]            # motor speed rad/s
    T = F * veh["Rw"] / veh["G"] + m["J_rotor"] * np.gradient(w)
    rpm = w / RPM
    Tmax = np.array([max_torque(cfg, max(r, 1.0)) for r in rpm])
    T_mot = np.clip(T, -Tmax, Tmax)         # beyond the envelope: friction brake / not reachable
    return dict(v=v, a=a, w=w, rpm=rpm, T_req=T, T=T_mot, dist_km=float(np.sum(v) / 1000))


def energy(cfg, loss_grid, d=None):
    """loss_grid: dict(rpm=array, torque=array, P=2-D array [rpm x torque], NaN outside envelope)."""
    d = d or demand(cfg)
    P = np.array(loss_grid["P"], dtype=float)
    # fill cells outside the envelope with the nearest feasible value along torque (needed only for interpolation)
    for i in range(P.shape[0]):
        row = P[i]
        good = np.where(np.isfinite(row))[0]
        if good.size:
            for j in range(row.size):
                if not np.isfinite(row[j]):
                    row[j] = row[good[np.argmin(np.abs(good - j))]]
    f = RegularGridInterpolator((loss_grid["rpm"], loss_grid["torque"]), P, bounds_error=False, fill_value=None)
    rpm = np.clip(d["rpm"], loss_grid["rpm"][0], loss_grid["rpm"][-1])
    tq = np.clip(np.abs(d["T"]), loss_grid["torque"][0], loss_grid["torque"][-1])
    Ploss = f(np.column_stack([rpm, tq]))
    Ploss[(d["v"] < 0.01) & (np.abs(d["T"]) < 1e-6)] = 0.0
    Ploss[d["v"] < 0.01] = 0.0
    Pmech = d["T"] * d["w"]
    E_loss = float(np.sum(Ploss)) / 3600.0                     # Wh
    E_trac = float(np.sum(np.clip(Pmech, 0, None))) / 3600.0   # Wh
    E_regen = float(np.sum(np.clip(Pmech, None, 0))) / 3600.0  # Wh (negative)
    E_bat = E_trac + E_regen + E_loss
    return dict(E_loss_Wh=E_loss, E_traction_Wh=E_trac, E_regen_Wh=E_regen, E_battery_Wh=E_bat,
                Wh_per_km=E_bat / d["dist_km"], loss_Wh_per_km=E_loss / d["dist_km"], dist_km=d["dist_km"])


def time_mean(cfg, grid, d=None, relative=False):
    """Time-average over moving samples of a map quantity (e.g. torque-ripple RMS).
    relative=True divides by max(|T|, 10 N m) before averaging (ripple as a fraction of demand)."""
    d = d or demand(cfg)
    P = np.array(grid["P"], dtype=float)
    for i in range(P.shape[0]):
        good = np.where(np.isfinite(P[i]))[0]
        for j in range(P.shape[1]):
            if not np.isfinite(P[i, j]) and good.size:
                P[i, j] = P[i, good[np.argmin(np.abs(good - j))]]
    f = RegularGridInterpolator((grid["rpm"], grid["torque"]), P, bounds_error=False, fill_value=None)
    m = d["v"] > 0.01
    rpm = np.clip(d["rpm"][m], grid["rpm"][0], grid["rpm"][-1])
    tq = np.clip(np.abs(d["T"][m]), grid["torque"][0], grid["torque"][-1])
    x = f(np.column_stack([rpm, tq]))
    if relative:
        x = x / np.maximum(np.abs(d["T"][m]), 10.0)
    return float(np.mean(x))
