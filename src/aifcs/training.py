"""Offline training.

(A) Manuscript method (Sec. 3.2): grid search of (lam_d, lam_q) per operating
    condition with J_train = a1*RMSE_iq + a2*T_ripple + a3*t_s + a4*D_iq (each term
    normalised by the (1,1) baseline of the same condition), label = argmin pair,
    features = instantaneous [e_d, e_q, w_m, TL_hat] of the best run, MLP 4-10-2 with
    sigmoid output scaled to [lam_min, lam_max].
    Two splits are reported: random sample split (as in the manuscript, leaks
    time-correlated samples) and split by operating condition (no leakage).

(B) Reformulation: lam_sw (switching-count penalty) scheduled on (w_m, TL_hat).
    Label per operating point = min total drive loss subject to
    T_rms <= ripple_limit_pct_of_peak * T_peak.
"""
import itertools
import json
import os
import numpy as np
from sklearn.neural_network import MLPRegressor

from . import core, scenarios as S, runner, metrics as M, ironloss as IL

RPM = 2 * np.pi / 60.0


def _logit(y):
    y = np.clip(y, 1e-4, 1 - 1e-4)
    return np.log(y / (1 - y))


def _export(mlp, xm, xs, lo, hi):
    return dict(W1=np.ascontiguousarray(mlp.coefs_[0].T), b1=mlp.intercepts_[0].copy(),
                W2=np.ascontiguousarray(mlp.coefs_[1].T), b2=mlp.intercepts_[1].copy(),
                xm=xm.astype(np.float64), xs=xs.astype(np.float64), lo=float(lo), hi=float(hi))


def save_nn(path, nn):
    np.savez(path, **{k: np.asarray(v) for k, v in nn.items()})


def load_nn(path):
    d = np.load(path)
    nn = {k: d[k] for k in d.files}
    nn["lo"] = float(nn["lo"])
    nn["hi"] = float(nn["hi"])
    return nn


# ---------------------------------------------------------------- (A) paper method
def paper_conditions():
    speeds = [500.0, 1500.0, 2500.0]
    iq_final = [500.0, 900.0]
    mism = [("nominal", {}), ("Rs+50%", {"Rs": 1.5}), ("L-20%", {"Ld": 0.8, "Lq": 0.8}),
            ("psi-10%", {"psi": 0.9})]
    conds = []
    for sp, iq1, (mn, mm) in itertools.product(speeds, iq_final, mism):
        conds.append(dict(rpm=sp, iq1=iq1, mname=mn, mm=mm))
    return conds


def eval_condition(cfg, Ts, cond, lam, seed, nsub):
    sc = S.step_current(cfg, Ts, rpm=cond["rpm"], iq0=200.0, iq1=cond["iq1"], t_step=0.01, T=0.04)
    r = runner.run(core.FCS, cfg, sc, mismatch=cond["mm"], seed=seed, Ts=Ts,
                   ctrl_overrides={"lam_d": lam[0], "lam_q": lam[1], "nsub": nsub})
    st = M.step_metrics(r, 0.01)
    tk = M.tracking(r, 0.01, 0.04)
    tr = M.torque_ripple(r, 0.025, 0.04)
    ia = M.iq_avg(r)
    k = int(0.025 / r["Ts"])
    D = float(np.max(np.abs(ia[k:] - cond["iq1"])))
    ts = st["t_settle_ms"]
    if not np.isfinite(ts):   # not settled within the window -> penalise with the window length
        ts = 1e3 * (0.04 - 0.01)
    return r, dict(RMSE_iq=tk["RMSE_iq"], T_rms=tr["T_rms"], t_s=ts, D_iq=D)


def train_paper_nn(cfg, Ts, outdir, seed=0):
    os.makedirs(outdir, exist_ok=True)
    pc = cfg["paper_nn"]
    grid = pc["lam_grid"]
    a = pc["alpha"]
    nsub = int(round(Ts / 0.25e-6))
    conds = paper_conditions()
    rows, X, Y, G = [], [], [], []
    for ci, c in enumerate(conds):
        res = {}
        for lam in itertools.product(grid, grid):
            r, m = eval_condition(cfg, Ts, c, lam, seed=1000 + ci, nsub=nsub)
            res[lam] = (r, m)
        base = res[(1.0, 1.0)][1]
        best, bestJ = None, np.inf
        for lam, (r, m) in res.items():
            J = (a[0] * m["RMSE_iq"] / base["RMSE_iq"] + a[1] * m["T_rms"] / base["T_rms"]
                 + a[2] * m["t_s"] / base["t_s"] + a[3] * m["D_iq"] / base["D_iq"])
            rows.append(dict(cond=ci, rpm=c["rpm"], iq1=c["iq1"], mismatch=c["mname"],
                             lam_d=lam[0], lam_q=lam[1], ratio=lam[0] / lam[1], J_train=J, **m))
            if J < bestJ - 1e-12:
                best, bestJ = lam, J
        r = res[best][0]
        log = r["log"]
        ks = np.arange(int(0.005 / Ts), log.shape[0], 4)
        feats = np.column_stack([log[ks, core.LIDREF] - log[ks, core.LIDM],
                                 log[ks, core.LIQREF] - log[ks, core.LIQM],
                                 log[ks, core.LWM], log[ks, core.LTLH]])
        X.append(feats)
        Y.append(np.tile(best, (ks.size, 1)))
        G.append(np.full(ks.size, ci))
        rows_best = [x for x in rows if x["cond"] == ci and x["lam_d"] == best[0] and x["lam_q"] == best[1]]
        rows_best[0]["is_label"] = True
    X, Y, G = np.vstack(X), np.vstack(Y), np.concatenate(G)
    lo, hi = pc["lam_min"], pc["lam_max"]
    T = _logit((Y - lo) / (hi - lo))
    xm, xs = X.mean(0), X.std(0) + 1e-9   # NOTE: statistics re-computed on train part below
    rng = np.random.default_rng(seed)
    report = {}

    def fit(idx_tr, idx_va, idx_te, tag):
        xm_, xs_ = X[idx_tr].mean(0), X[idx_tr].std(0) + 1e-9
        Xn = (X - xm_) / xs_
        mlp = MLPRegressor(hidden_layer_sizes=(pc["hidden"],), activation="tanh", solver="adam",
                           learning_rate_init=pc["lr"], max_iter=300, random_state=seed)
        mlp.fit(Xn[idx_tr], T[idx_tr])
        nn = _export(mlp, xm_, xs_, lo, hi)
        pred = lo + (hi - lo) / (1 + np.exp(-mlp.predict(Xn)))
        mse = lambda I: float(np.mean((pred[I] - Y[I]) ** 2))
        report[tag] = dict(n_train=int(idx_tr.size), n_val=int(idx_va.size), n_test=int(idx_te.size),
                           mse_train=mse(idx_tr), mse_val=mse(idx_va), mse_test=mse(idx_te),
                           epochs=int(mlp.n_iter_))
        return nn

    # random sample split (manuscript)
    perm = rng.permutation(X.shape[0])
    n1, n2 = int(0.7 * len(perm)), int(0.85 * len(perm))
    fit(perm[:n1], perm[n1:n2], perm[n2:], "random_sample_split")
    # split by condition (used for deployment)
    cperm = rng.permutation(len(conds))
    c_tr, c_va, c_te = cperm[:17], cperm[17:20], cperm[20:]
    I = lambda cs: np.where(np.isin(G, cs))[0]
    nn = fit(I(c_tr), I(c_va), I(c_te), "condition_split")
    report["condition_split"]["test_conditions"] = [conds[i]["mname"] + f"@{conds[i]['rpm']:.0f}rpm,iq{conds[i]['iq1']:.0f}" for i in c_te]
    # best single fixed pair over the training conditions (strong fixed baseline)
    Jsum = {}
    for x in rows:
        if x["cond"] in c_tr:
            Jsum[(x["lam_d"], x["lam_q"])] = Jsum.get((x["lam_d"], x["lam_q"]), 0.0) + x["J_train"]
    best_fixed = min(Jsum, key=Jsum.get)
    report["best_fixed_pair"] = list(best_fixed)
    report["label_ratio_counts"] = {}
    for ci in range(len(conds)):
        lab = [x for x in rows if x["cond"] == ci and x.get("is_label")][0]
        key = f"{lab['ratio']:.3g}"
        report["label_ratio_counts"][key] = report["label_ratio_counts"].get(key, 0) + 1
    save_nn(os.path.join(outdir, "paper_nn.npz"), nn)
    import csv
    with open(os.path.join(outdir, "paper_grid_search.csv"), "w", newline="") as f:
        keys = ["cond", "rpm", "iq1", "mismatch", "lam_d", "lam_q", "ratio", "J_train",
                "RMSE_iq", "T_rms", "t_s", "D_iq", "is_label"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for x in rows:
            w.writerow({k: x.get(k, "") for k in keys})
    with open(os.path.join(outdir, "paper_nn_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    return nn, best_fixed, report


# ---------------------------------------------------------------- (B) lam_sw scheduling
def sw_point(cfg, Ts, rpm, torque, lam_sw, seed, ctrl=core.FCS, nn=None, lut=None):
    nsub = int(round(Ts / 0.25e-6))
    sc = S.steady(cfg, Ts, rpm, torque, T=0.03)
    if sc is None:
        return None
    r = runner.run(ctrl, cfg, sc, nn=nn, lut=lut, seed=seed, Ts=Ts,
                   ctrl_overrides={"lam_sw": lam_sw, "nsub": nsub})
    L = M.losses(r)
    tr = M.torque_ripple(r, 0.01, 0.03)
    # iron-loss waveform statistics: extra columns only, they do not enter the label selection below
    ws = {"ws_" + k: v for k, v in IL.waveform_stats(r, 0.01, None, IL.IronParams.from_cfg(cfg)).items()}
    return dict(rpm=rpm, torque=torque, lam_sw=lam_sw, fsw=M.fsw(r), T_rms=tr["T_rms"],
                T_mean=tr["T_mean"], P_sw=L["P_sw"], P_cond=L["P_cond"], P_cu=L["P_cu"],
                P_loss=L["P_sw"] + L["P_cond"] + L["P_cu"], P_mech=L["P_mech"], **ws)


def train_sw_scheduler(cfg, Ts, outdir, seed=0, rule="abs"):
    """rule='abs': T_rms <= pct * T_peak (absolute NVH limit);
       rule='rel': T_rms <= pct * operating torque (relative limit)."""
    os.makedirs(outdir, exist_ok=True)
    ec = cfg["ext_sw"]
    Tlim = ec["ripple_limit_pct_of_peak"] / 100 * 430.0
    speeds = [500.0, 1500.0, 3000.0, 4500.0, 6000.0, 8000.0, 9500.0]
    torques = [25.0, 50.0, 100.0, 200.0, 300.0, 400.0]
    rows, table = [], np.full((len(speeds), len(torques)), np.nan)
    for i, sp in enumerate(speeds):
        for j, tq in enumerate(torques):
            cand = [sw_point(cfg, Ts, sp, tq, lam, seed=2000 + 10 * i + j) for lam in ec["lam_sw_grid"]]
            if cand[0] is None:      # outside the torque-speed envelope
                continue
            rows += cand
            lim = Tlim if rule == "abs" else ec["ripple_limit_pct_of_peak"] / 100 * tq
            feas = [c for c in cand if c["T_rms"] <= lim]
            pick = min(feas, key=lambda c: c["P_loss"]) if feas else min(cand, key=lambda c: c["T_rms"])
            lam = max(pick["lam_sw"], 10 ** ec["log_lo"])
            table[i, j] = np.log10(lam)
            pick["is_label"] = True
    for i in range(table.shape[0]):           # fill infeasible cells (never reached) with nearest feasible
        good = np.where(np.isfinite(table[i]))[0]
        for j in range(table.shape[1]):
            if not np.isfinite(table[i, j]):
                table[i, j] = table[i, good[np.argmin(np.abs(good - j))]]
    feas_pts = {(x["rpm"], x["torque"]) for x in rows}
    Kt = 1.5 * cfg["motor"]["p"] * cfg["motor"]["psi_f"]
    lut = dict(x=np.array(speeds) * RPM, y=np.array(torques), v=table)
    # NN 2-6-1 on the grid (inputs: w_m [rad/s], TL_hat [N m] ~ torque in steady state)
    pts = [(i, j) for i, sp in enumerate(speeds) for j, tq in enumerate(torques) if (sp, tq) in feas_pts]
    Xg = np.array([[speeds[i] * RPM, torques[j]] for i, j in pts])
    yg = (np.array([table[i, j] for i, j in pts]) - ec["log_lo"]) / (ec["log_hi"] - ec["log_lo"])
    xm, xs = Xg.mean(0), Xg.std(0)
    mlp = MLPRegressor(hidden_layer_sizes=(6,), activation="tanh", solver="lbfgs", max_iter=5000,
                       random_state=seed, alpha=1e-3)
    mlp.fit((Xg - xm) / xs, _logit(yg))
    nn = _export(mlp, xm, xs, ec["log_lo"], ec["log_hi"])
    nn["W2"] = nn["W2"].reshape(1, -1)
    save_nn(os.path.join(outdir, "sw_nn.npz"), nn)
    np.savez(os.path.join(outdir, "sw_lut.npz"), **lut)
    # single best fixed lam_sw: minimum total loss among values feasible at every grid point
    fixed = {}
    for lam in ec["lam_sw_grid"]:
        rr = [x for x in rows if x["lam_sw"] == lam]
        fixed[lam] = (all(x["T_rms"] <= (Tlim if rule == "abs" else ec["ripple_limit_pct_of_peak"] / 100 * x["torque"]) for x in rr),
                      sum(x["P_loss"] for x in rr))
    feas_fixed = [l for l, (f, _) in fixed.items() if f]
    best_fixed = min(feas_fixed, key=lambda l: fixed[l][1]) if feas_fixed else 0.0
    import csv
    with open(os.path.join(outdir, "sw_grid_search.csv"), "w", newline="") as f:
        keys = list(rows[0].keys()) + ["is_label"]
        keys = list(dict.fromkeys(keys))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for x in rows:
            w.writerow({k: x.get(k, "") for k in keys})
    rep = dict(rule=rule, T_limit_Nm=Tlim if rule == "abs" else f"{ec['ripple_limit_pct_of_peak']}% of operating torque", best_fixed_lam_sw=best_fixed, lut_log10=table.tolist(),
               speeds_rpm=speeds, torques_Nm=torques, Kt=Kt)
    with open(os.path.join(outdir, "sw_scheduler_report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    return nn, lut, best_fixed, rep
