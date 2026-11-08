"""Variable-switching-frequency (VSF) PWM baselines and ripple-constrained scheduling.

Why this module exists
  A fixed-carrier PWM baseline is an unfair comparator for a switching-penalised FCS-MPCC
  (whose average switching frequency is operating-point dependent). Here every method gets its
  own tunable knob and the knob is scheduled per operating point under the SAME torque-ripple
  constraint:
    PI-SVPWM, PI-DPWM1, PI-DPWM-I : carrier frequency (= 1 / control period Ts) from CARRIERS_HZ
    FCS-MPCC                      : lam_sw (at Ts = 10 us), optionally also Ts in {10, 20, 50} us
  Carrier scheduling is realised OFFLINE as a look-up table (LUT) over (speed, torque): each map
  point is simulated at its own fixed carrier; Ts is never changed inside a run.

PI bandwidth rule (params.build): wc = 2 pi min(800 Hz, f_carrier / 12), applied to every carrier.

Maps are steady-state runs of length max(30 ms, 1 electrical period + 10 ms); the ripple window is
[10 ms, T_end] (>= 1 electrical period, so the DPWM clamp sequence is covered at low speed); losses
are E / T over the whole run, like the maps of reproduce.py section 6.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from . import ironloss
from . import core, scenarios as S, runner, metrics as M, wltc

TRACK_REL, TRACK_ABS = 0.10, 5.0                     # torque-tracking failure: |bias| > max(10 %, 5 N m)
H_BASE = 0.25e-6                                     # plant sub-step (s)
CARRIER_TS_US = (1000.0, 2000.0 / 3.0, 500.0, 400.0, 275.0, 200.0, 400.0 / 3.0, 100.0, 200.0 / 3.0, 50.0)
# carrier set 1, 1.5, 2, 2.5, 3.636, 5, 7.5, 10, 15, 20 kHz (1-2 kHz added so that the PWM baselines are not
# artificially stopped above the average switching frequency that FCS-MPCC reaches at light load)
CARRIERS_HZ = tuple(1e6 / t for t in CARRIER_TS_US)
FCS_TS_US = (10.0, 20.0, 50.0)
LAM_GRID = (0.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)

PWM_FAMILIES = [("PI-SVPWM", core.PI), ("PI-DPWM1", core.DPWM1), ("PI-DPWM-I", core.DPWMI)]


def carrier_label(fc):
    return f"{fc / 1e3:.4g} kHz"


def settings_list():
    """All (family, knob) settings. Each: dict(family, label, ctrl, Ts, ov, knob)."""
    out = []
    for fam, cid in PWM_FAMILIES:
        for tus, fc in zip(CARRIER_TS_US, CARRIERS_HZ):
            out.append(dict(family=fam, label=carrier_label(fc), ctrl=cid, Ts=tus * 1e-6, ov={}, knob=fc))
    for tus in FCS_TS_US:
        for lam in LAM_GRID:
            out.append(dict(family=f"FCS Ts={tus:g}us", label=f"lam_sw={lam:g}", ctrl=core.FCS, Ts=tus * 1e-6,
                            ov={"lam_sw": lam}, knob=lam))
    return out


# scheduled methods = which families the schedule may pick from
METHODS = {
    "PI-SVPWM VSF": ["PI-SVPWM"],
    "PI-DPWM1 VSF": ["PI-DPWM1"],
    "PI-DPWM-I VSF": ["PI-DPWM-I"],
    "FCS lam_sw sched (Ts=10us)": ["FCS Ts=10us"],
    "FCS (Ts, lam_sw) sched": ["FCS Ts=10us", "FCS Ts=20us", "FCS Ts=50us"],
}


def sim_time(cfg, rpm):
    fe = cfg["motor"]["p"] * rpm / 60.0
    return max(0.03, 1.0 / fe + 0.01)


def point_run(cfg, st, rpm, torque, seed):
    """One steady-state map run for a setting. Returns (dict of metrics, run dict) or (None, None)."""
    T = sim_time(cfg, rpm)
    sc = S.steady(cfg, st["Ts"], rpm, torque, T=T)
    if sc is None:
        return None, None
    ov = dict(st["ov"], nsub=int(round(st["Ts"] / H_BASE)))
    r = runner.run(st["ctrl"], cfg, sc, seed=seed, Ts=st["Ts"], ctrl_overrides=ov)
    L = M.losses(r)
    tr = M.torque_ripple(r, 0.01, T)
    fe = ironloss.iron_loss(r, cfg, t0=0.01, t1=T)
    return dict(P_cu=L["P_cu"], P_cond=L["P_cond"], P_sw=L["P_sw"], P_mech=L["P_mech"], T_rms=tr["T_rms"],
                P_fe=fe["P_fe"], P_fe_harm=fe["P_harm"],
                T_mean=tr["T_mean"], f_sw_Hz=M.fsw(r), ok=bool(r["ok"])), r


def build_maps(cfg, rpm_grid, torque_grid, seeds, record=None, tag="S10_vsf_map", settings=None, seed_offset=0):
    """Raw rows (one per setting x node x seed)."""
    rows = []
    for st in (settings or settings_list()):
        for rpm in rpm_grid:
            for tq in torque_grid:
                for sd in seeds:
                    m, r = point_run(cfg, st, rpm, tq, sd + seed_offset)
                    if m is None:
                        break
                    if record:
                        record(r, tag)
                    rows.append(dict(family=st["family"], label=st["label"], knob=st["knob"], Ts_us=st["Ts"] * 1e6,
                                     rpm=rpm, torque=tq, seed=sd + seed_offset, **m))
    return pd.DataFrame(rows)


def average_maps(raw):
    g = raw.groupby(["family", "label", "knob", "Ts_us", "rpm", "torque"], sort=False)
    m = g[["P_cu", "P_cond", "P_sw", "P_mech", "P_fe", "P_fe_harm", "T_rms", "T_mean", "f_sw_Hz"]].mean()
    m["T_rms_std"] = g["T_rms"].std(ddof=1)
    m["ok"] = g["ok"].all()
    m = m.reset_index()
    m["T_bias"] = m.T_mean - m.torque
    # a setting that does not deliver the requested torque (unstable / badly biased) is never feasible,
    # whatever its ripple (std) is
    m["track_fail"] = m.T_bias.abs() > np.maximum(TRACK_REL * m.torque, TRACK_ABS)
    return m


# --------------------------------------------------------------------------- limits and selection
def limit_abs(value):
    return lambda rpm, tq: value


def limit_rel(pct, floor):
    return lambda rpm, tq: max(pct / 100.0 * tq, floor)


def ploss(df, k_sw=1.0, k_cond=1.0, k_fe=0.0):
    """Drive loss [W]. k_fe=1 adds the stator iron loss with PWM harmonics (ironloss.py, column P_fe)."""
    out = df.P_cu + k_cond * df.P_cond + k_sw * df.P_sw
    if k_fe:
        out = out + k_fe * df.P_fe
    return out


def select(dfm, fams, limit_fn, k_sw=1.0, k_cond=1.0, k_fe=0.0):
    """Per node: setting of the allowed families with minimum drive loss s.t. T_rms <= limit.
    If nothing is feasible the minimum-ripple setting is used and flagged (feasible=False)."""
    sub = dfm[dfm.family.isin(fams)].copy()
    sub["P_loss"] = ploss(sub, k_sw, k_cond, k_fe)
    rows = []
    for (rpm, tq), g in sub.groupby(["rpm", "torque"], sort=False):
        lim = limit_fn(rpm, tq)
        f = g[(g.T_rms <= lim) & (~g.track_fail)]
        if len(f):
            pick, feas = f.loc[f.P_loss.idxmin()], True
        else:
            gg = g[~g.track_fail] if (~g.track_fail).any() else g
            pick, feas = gg.loc[gg.T_rms.idxmin()], False
        rows.append(dict(rpm=rpm, torque=tq, family=pick.family, label=pick.label, knob=pick.knob,
                         Ts_us=pick.Ts_us, limit=lim, feasible=feas, P_loss=pick.P_loss, T_rms=pick.T_rms))
    return pd.DataFrame(rows)


def fixed_feasible(dfm, fams, limit_fn):
    """Settings (family, label) of the allowed families that meet the limit and track the torque at EVERY
    map node, i.e. the candidates for one single, unscheduled setting."""
    sub = dfm[dfm.family.isin(fams)].copy()
    sub["ok_lim"] = [(r.T_rms <= limit_fn(r.rpm, r.torque)) and (not r.track_fail) for r in sub.itertuples()]
    return [(fam, lab) for (fam, lab), g in sub.groupby(["family", "label"], sort=False) if g.ok_lim.all()]


# --------------------------------------------------------------------------- LUT lookup
def to_grid(df, col, rpm_grid, torque_grid):
    g = np.full((len(rpm_grid), len(torque_grid)), np.nan)
    ri = {v: i for i, v in enumerate(rpm_grid)}
    ti = {v: j for j, v in enumerate(torque_grid)}
    for r in df.itertuples():
        g[ri[r.rpm], ti[r.torque]] = getattr(r, col)
    return g


def fill_along_torque(g):
    """Fill NaN nodes (outside the envelope) with the nearest valid node along torque
    (same rule as wltc.energy)."""
    g = np.array(g, dtype=float)
    for i in range(g.shape[0]):
        good = np.where(np.isfinite(g[i]))[0]
        for j in range(g.shape[1]):
            if not np.isfinite(g[i, j]) and good.size:
                g[i, j] = g[i, good[np.argmin(np.abs(good - j))]]
    return g


def nearest_node(rpm_grid, torque_grid, rpm, torque):
    """Nearest node in grid-index space (the grids are non-uniform). Returns integer index arrays."""
    rpm_grid, torque_grid = np.asarray(rpm_grid, float), np.asarray(torque_grid, float)
    fi = np.interp(np.asarray(rpm, float), rpm_grid, np.arange(rpm_grid.size))
    fj = np.interp(np.asarray(torque, float), torque_grid, np.arange(torque_grid.size))
    return np.floor(fi + 0.5).astype(int), np.floor(fj + 0.5).astype(int)


def make_lut(sel, rpm_grid, torque_grid, setting_ids):
    """Selected-setting index per node (NaN-filled along torque)."""
    idx = {sid: k for k, sid in enumerate(setting_ids)}
    df = sel.copy()
    df["sid"] = [idx[(f, l)] for f, l in zip(df.family, df.label)]
    return fill_along_torque(to_grid(df, "sid", rpm_grid, torque_grid)).astype(int)


def lookup(lut_idx, setting_ids, rpm_grid, torque_grid, rpm, torque):
    """Deployable LUT: returns the setting id(s) (family, label) of the nearest node."""
    i, j = nearest_node(rpm_grid, torque_grid, rpm, torque)
    k = lut_idx[i, j]
    if np.ndim(k) == 0:
        return setting_ids[int(k)]
    return [setting_ids[int(x)] for x in k]


def carrier_lookup(lut_idx, setting_ids, rpm_grid, torque_grid, rpm, torque):
    """Carrier frequency (Hz) from a PWM-family LUT for one operating point; always a member of CARRIERS_HZ."""
    fam, lab = lookup(lut_idx, setting_ids, rpm_grid, torque_grid, rpm, torque)
    return CARRIERS_HZ[[carrier_label(f) for f in CARRIERS_HZ].index(lab)]


# --------------------------------------------------------------------------- WLTC evaluation
class WltcEval:
    """Quasi-static WLTC evaluation. Same assumptions as wltc.py (gear efficiency 1, symmetric
    motoring/regeneration, zero loss at standstill, no iron/mechanical loss)."""

    def __init__(self, cfg, rpm_grid, torque_grid):
        self.cfg, self.rg, self.tg = cfg, np.asarray(rpm_grid, float), np.asarray(torque_grid, float)
        self.d = wltc.demand(cfg)
        d = self.d
        self.moving = d["v"] > 0.01
        self.rpm = np.clip(d["rpm"], self.rg[0], self.rg[-1])
        self.tq = np.clip(np.abs(d["T"]), self.tg[0], self.tg[-1])
        self.Tabs = np.abs(d["T"])
        self.dist = d["dist_km"]
        self.pts = np.column_stack([self.rpm, self.tq])

    def series(self, grid):
        f = RegularGridInterpolator((self.rg, self.tg), fill_along_torque(grid), bounds_error=False, fill_value=None)
        return f(self.pts)

    def nearest_idx(self, lut_idx):
        i, j = nearest_node(self.rg, self.tg, self.rpm, self.tq)
        return lut_idx[i, j]

    def summarise(self, Ploss, Trms, fsw, limit_fn=None):
        P = np.array(Ploss, float)
        P[~self.moving] = 0.0
        loss = float(P.sum() / 3600.0 / self.dist)
        T = Trms[self.moving]
        out = dict(loss_Wh_per_km=loss, ripple_mean_Nm=float(T.mean()),
                   ripple_rel_mean_pct=float(100 * np.mean(T / np.maximum(self.Tabs[self.moving], 10.0))),
                   fsw_mean_kHz=float(np.mean(fsw[self.moving]) / 1e3))
        if limit_fn is not None:
            lim = np.array([limit_fn(r, t) for r, t in zip(self.d["rpm"][self.moving], self.Tabs[self.moving])])
            out["viol_time_frac"] = float(np.mean(T > lim))
        return out


def setting_series(ev, dfm, setting_ids, k_sw=1.0, k_cond=1.0, k_fe=0.0):
    """Per setting: WLTC time series of drive loss, ripple, f_sw (interpolating the setting's own map) and a
    boolean 'fails torque tracking at this operating point' (any neighbouring node fails)."""
    P, T, F, B = [], [], [], []
    for fam, lab in setting_ids:
        g = dfm[(dfm.family == fam) & (dfm.label == lab)]
        P.append(ev.series(to_grid(g.assign(P_loss=ploss(g, k_sw, k_cond, k_fe)), "P_loss", ev.rg, ev.tg)))
        T.append(ev.series(to_grid(g, "T_rms", ev.rg, ev.tg)))
        F.append(ev.series(to_grid(g, "f_sw_Hz", ev.rg, ev.tg)))
        B.append(ev.series(to_grid(g.assign(tf=g.track_fail.astype(float)), "tf", ev.rg, ev.tg)) > 1e-9)
    return np.array(P), np.array(T), np.array(F), np.array(B)


def limit_series(ev, limit_fn):
    return np.array([limit_fn(r, t) for r, t in zip(ev.d["rpm"], ev.Tabs)])


def eval_lut(ev, ser, lut_idx, limit_fn=None):
    """Deployable LUT: the nearest map node picks the setting; loss / ripple / f_sw are those of that
    setting's own map interpolated at the true operating point."""
    P, T, F, B = ser
    k = ev.nearest_idx(lut_idx)
    n = np.arange(k.size)
    return ev.summarise(P[k, n], T[k, n], F[k, n], limit_fn)


def eval_pointwise(ev, ser, allowed, limit_fn):
    """Ideal (continuous) schedule: at every WLTC sample choose, among the allowed settings (indices),
    the one with minimum interpolated drive loss whose interpolated ripple meets the limit and that tracks the
    torque; if none does, the minimum-ripple one (counted as a violation). Returns (summary, chosen indices)."""
    P, T, F, B = (a[allowed] for a in ser)
    lim = limit_series(ev, limit_fn)[None, :]
    feas = (T <= lim) & (~B)
    Pm = np.where(feas, P, np.inf)
    kf = np.argmin(Pm, axis=0)
    anyf = feas.any(axis=0)
    Tm = np.where(~B, T, np.inf)
    Tm = np.where(np.isfinite(Tm).any(axis=0)[None, :], Tm, T)
    kmin = np.argmin(Tm, axis=0)
    k = np.where(anyf, kf, kmin)
    n = np.arange(k.size)
    out = ev.summarise(P[k, n], T[k, n], F[k, n], limit_fn)
    out["infeasible_time_frac"] = float(np.mean((~anyf)[ev.moving]))
    return out, np.asarray(allowed)[k]
