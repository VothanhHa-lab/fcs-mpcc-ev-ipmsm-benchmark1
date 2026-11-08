"""Performance indices, all computed from the simulated (true-plant) fine log.

Definitions (also in README):
  iq_avg      : true i_q averaged over each control period Ts (removes intra-period ripple)
  overshoot   : max(iq_avg) - final reference after the step, >= 0  [A]
  t_settle    : time after the step until iq_avg stays within +-band (default 5 % of the step) [s]
  RMSE / MAE  : of (i_q_ref - i_q_true) on the fine grid, over a window [A]
  T_pp / T_rms: peak-to-peak and RMS (std) of true electromagnetic torque over a steady window [N m]
  f_sw        : average device switching frequency = commutations per phase / (2 T) [Hz]
  THD_i       : phase-a current THD up to 50 kHz over an integer number of electrical periods
"""
import numpy as np
from . import core

RPM = 2 * np.pi / 60.0


def _ref_on_fine(res, col):
    log, fine = res["log"], res["fine"]
    k = np.minimum((fine[:, core.FT] / res["Ts"] + 1e-9).astype(int), log.shape[0] - 1)
    return log[k, col]


def iq_avg(res):
    fine = res["fine"]
    per = int(res["cp"][1] // res["cp"][15])
    n = fine.shape[0] // per
    return fine[:n * per, core.FIQ].reshape(n, per).mean(axis=1)


def window(res, t0, t1):
    t = res["fine"][:, core.FT]
    return (t >= t0) & (t < t1)


def tracking(res, t0, t1):
    w = window(res, t0, t1)
    e_q = _ref_on_fine(res, core.LIQREF)[w] - res["fine"][w, core.FIQ]
    e_d = _ref_on_fine(res, core.LIDREF)[w] - res["fine"][w, core.FID]
    return dict(RMSE_iq=float(np.sqrt(np.mean(e_q ** 2))), MAE_iq=float(np.mean(np.abs(e_q))),
                RMSE_id=float(np.sqrt(np.mean(e_d ** 2))))


def step_metrics(res, t_step, band_frac=0.05):
    ia = iq_avg(res)
    Ts = res["Ts"]
    ref = res["log"][:, core.LIQREF]
    k0 = int(round(t_step / Ts))
    final = ref[-1]
    step = final - ref[max(k0 - 1, 0)]
    band = abs(step) * band_frac
    post = ia[k0 + 1:]   # +1: first period is the computation delay
    os_ = max(0.0, float(np.max(np.sign(step) * (post - final))))
    outside = np.where(np.abs(post - final) > band)[0]
    if outside.size and outside[-1] >= post.size - 1 - int(0.002 / Ts):
        ts = np.nan   # still outside the band at the end of the window: not settled
    else:
        ts = (outside[-1] + 2) * Ts if outside.size else Ts
    return dict(overshoot_A=os_, t_settle_ms=1e3 * ts, band_A=band)


def torque_ripple(res, t0, t1):
    w = window(res, t0, t1)
    Te = res["fine"][w, core.FTE]
    return dict(T_mean=float(Te.mean()), T_pp=float(Te.max() - Te.min()), T_rms=float(Te.std()))


def fsw(res):
    return float(np.mean(res["ncomm"]) / (2.0 * res["T"]))


def thd(res, t0, t1, fmax=50e3):
    fine = res["fine"]
    t = fine[:, core.FT]
    w_e = res["pp"][4] * float(np.mean(fine[:, core.FW]))
    if w_e <= 0:
        return np.nan
    Te = 2 * np.pi / w_e
    nper = int((t1 - t0) / Te)
    if nper < 1:
        return np.nan
    m = (t >= t1 - nper * Te) & (t < t1)
    x = fine[m, core.FIA]
    dt = t[1] - t[0]
    X = np.abs(np.fft.rfft(x)) / x.size
    f = np.fft.rfftfreq(x.size, dt)
    k1 = int(round(nper))  # fundamental bin
    fund = X[k1]
    harm = X[(f > 1.5 * w_e / (2 * np.pi)) & (f <= fmax)]
    return float(np.sqrt(np.sum(harm ** 2)) / fund)


def losses(res):
    E = res["E"]
    T = res["T"]
    return dict(P_cu=E[0] / T, P_cond=E[1] / T, P_sw=E[2] / T, P_mech=E[3] / T, P_elec=E[4] / T)


def speed_metrics(res, t_event, band_rpm=2.0):
    fine = res["fine"]
    t = fine[:, core.FT]
    w = fine[:, core.FW] / RPM
    wref = _ref_on_fine(res, core.LWREF) / RPM
    m = t >= t_event
    err = wref[m] - w[m]
    dev = float(np.max(np.abs(err)))
    out = np.where(np.abs(err) > band_rpm)[0]
    trec = (t[m][out[-1]] - t_event) if out.size else 0.0
    return dict(speed_dev_rpm=dev, t_recover_ms=1e3 * trec)
