"""Reproduce every figure and table of the study from scratch.

    python3 scripts/reproduce.py            # full run
    python3 scripts/reproduce.py --quick    # fewer seeds / Monte-Carlo samples (pipeline check)

Outputs: results/figures/*.pdf|png, results/tables/*.csv, results/raw/*.npz,
         results/training_*/, results/run_manifest.json
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numba  # noqa: E402
import sklearn  # noqa: E402
from aifcs import core, params, scenarios as S, runner, metrics as M, training, plotting as P  # noqa: E402
from aifcs import ironloss as IL  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()

cfg = params.load_config()
IP = IL.IronParams.from_cfg(cfg)      # stator iron-loss model (nominal coefficients)
ROOT = params.ROOT
RES = os.path.join(ROOT, "results")
FIG, TAB, RAW = (os.path.join(RES, d) for d in ("figures", "tables", "raw"))
for d in (FIG, TAB, RAW):
    os.makedirs(d, exist_ok=True)
SEEDS = [0, 1] if args.quick else [0, 1, 2, 3, 4]
N_MC = 6 if args.quick else 40
TS = cfg["control"]["Ts"]           # 50 us, manuscript
TS_FAST = 10e-6                      # FCS sampling giving ~10 kHz average switching
manifest = dict(started=datetime.now().isoformat(timespec="seconds"), quick=args.quick,
                python=sys.version.split()[0], numpy=np.__version__, numba=numba.__version__,
                sklearn=sklearn.__version__, platform=platform.platform(),
                config_sha1=hashlib.sha1(open(os.path.join(ROOT, "configs", "default.json"), "rb").read()).hexdigest(),
                seeds=SEEDS, n_mc=N_MC, runs=[])
T0 = time.time()


def record(r, tag):
    manifest["runs"].append(dict(tag=tag, ctrl=r["name"], scenario=r["scenario"], seed=r["seed"],
                                 mismatch=r["mismatch"], Ts=r["Ts"], ok=bool(r["ok"]), wall_s=round(r["wall"], 4)))


def agg(df, by, cols):
    g = df.groupby(by, sort=False)[cols]
    m, s = g.mean(), g.std(ddof=1)
    out = m.copy().astype(object)
    for c in cols:
        out[c] = [f"{a:.3g} ± {b:.2g}" for a, b in zip(m[c], s[c])]
    return out.reset_index(), m.reset_index()


# ------------------------------------------------------------------ 0. sanity checks
print("[0] sanity checks")
import subprocess  # noqa: E402
subprocess.run([sys.executable, os.path.join(ROOT, "tests", "test_sanity.py")], check=False)

# ------------------------------------------------------------------ 1. training
print("[1] training manuscript NN (Ts = 50 us)")
nn_paper, best_pair, rep_paper = training.train_paper_nn(cfg, TS, os.path.join(RES, "training_Ts50us"))
print("    best fixed pair", best_pair, "| test MSE (cond split)", rep_paper["condition_split"]["mse_test"])
print("[1] training lam_sw scheduler (Ts = 10 us)")
nn_sw, lut_sw, best_sw, rep_sw = training.train_sw_scheduler(cfg, TS_FAST, os.path.join(RES, "training_sw_Ts10us"))
print("    best fixed lam_sw", best_sw)

CTRLS = [("PI-SVPWM", core.PI, {}, None),
         ("Deadbeat-SVPWM", core.DB, {}, None),
         ("FCS-MPCC (1,1)", core.FCS, {"lam_d": 1.0, "lam_q": 1.0}, None),
         ("FCS-MPCC best fixed", core.FCS, {"lam_d": best_pair[0], "lam_q": best_pair[1]}, None),
         ("AI-FCS-MPCC (paper)", core.FCS_NN, {}, nn_paper)]


def run_set(sc, tag, seeds=SEEDS, mismatch=None, Ts=None, ctrls=CTRLS, keep_seed0=True):
    out = {}
    for lab, cid, ov, nn in ctrls:
        for sd in seeds:
            r = runner.run(cid, cfg, sc, nn=nn, mismatch=mismatch, seed=sd, Ts=Ts, ctrl_overrides=ov)
            record(r, tag)
            out[(lab, sd)] = r
            if keep_seed0 and sd == seeds[0]:
                np.savez_compressed(os.path.join(RAW, f"{tag}_{lab.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}_seed{sd}.npz"),
                                    log=r["log"], fine=r["fine"], ncomm=r["ncomm"], E=r["E"])
    return out


# ------------------------------------------------------------------ 2. manuscript scenarios @ Ts = 50 us
print("[2] S1 step current (bench, 1000 rpm, 200 -> 600 A)")
sc1 = S.step_current(cfg, TS)
R1 = run_set(sc1, "S1")
rows = []
for (lab, sd), r in R1.items():
    rows.append(dict(controller=lab, seed=sd, **M.step_metrics(r, sc1["t_event"]),
                     **M.tracking(r, 0.025, 0.06), **M.torque_ripple(r, 0.045, 0.06), f_sw_Hz=M.fsw(r)))
d1 = pd.DataFrame(rows)
d1.to_csv(os.path.join(TAB, "S1_step_all_seeds.csv"), index=False)
cols1 = ["overshoot_A", "t_settle_ms", "RMSE_iq", "MAE_iq", "T_rms", "T_pp", "f_sw_Hz"]
a1, _ = agg(d1, "controller", cols1)
a1.to_csv(os.path.join(TAB, "Table5_S1_step.csv"), index=False)
P.step_panels([R1[(l, SEEDS[0])] for l, *_ in CTRLS], [l for l, *_ in CTRLS], FIG, "Fig3_S1_step_current")

print("[2] S2 road-grade step 0 -> 10 % at 1000 rpm (vehicle inertia)")
sc2 = S.grade_step(cfg, TS)
R2 = run_set(sc2, "S2")
rows = []
for (lab, sd), r in R2.items():
    ia = M.iq_avg(r)
    k0 = int(sc2["t_event"] / TS)
    D = float(np.max(np.abs(r["log"][k0:ia.size, core.LIQREF] - ia[k0:])))
    rows.append(dict(controller=lab, seed=sd, **M.speed_metrics(r, sc2["t_event"]),
                     **M.tracking(r, sc2["t_event"], sc2["T"]), D_iq_A=D,
                     **M.torque_ripple(r, 0.5, 0.6), f_sw_Hz=M.fsw(r)))
d2 = pd.DataFrame(rows)
d2.to_csv(os.path.join(TAB, "S2_grade_all_seeds.csv"), index=False)
cols2 = ["speed_dev_rpm", "t_recover_ms", "RMSE_iq", "D_iq_A", "T_rms", "f_sw_Hz"]
a2, _ = agg(d2, "controller", cols2)
a2.to_csv(os.path.join(TAB, "Table6_S2_grade.csv"), index=False)
P.vehicle_panels([R2[(l, SEEDS[0])] for l, *_ in CTRLS], [l for l, *_ in CTRLS], FIG, "Fig4_S2_grade_step")

print("[2] S3 speed step 500 -> 1200 rpm (vehicle inertia)")
sc3 = S.speed_step(cfg, TS)
R3 = run_set(sc3, "S3")
rows = []
for (lab, sd), r in R3.items():
    f = r["fine"]
    wr = f[:, core.FW] * 60 / (2 * np.pi)
    rows.append(dict(controller=lab, seed=sd, speed_overshoot_rpm=float(max(0.0, wr.max() - 1200)),
                     t_settle_ms=M.speed_metrics(r, sc3["t_event"])["t_recover_ms"],
                     **M.tracking(r, sc3["t_event"], sc3["T"]), **M.torque_ripple(r, 0.9, 1.0),
                     f_sw_Hz=M.fsw(r)))
d3 = pd.DataFrame(rows)
d3.to_csv(os.path.join(TAB, "S3_speed_all_seeds.csv"), index=False)
cols3 = ["speed_overshoot_rpm", "t_settle_ms", "RMSE_iq", "T_rms", "f_sw_Hz"]
a3, _ = agg(d3, "controller", cols3)
a3.to_csv(os.path.join(TAB, "Table7_S3_speed.csv"), index=False)
P.vehicle_panels([R3[(l, SEEDS[0])] for l, *_ in CTRLS], [l for l, *_ in CTRLS], FIG, "Fig5_S3_speed_step")

print("[2] S4 steady torque ripple (bench, 1500 rpm, 200 N m)")
sc4 = S.steady(cfg, TS, 1500.0, 200.0, T=0.06)
R4 = run_set(sc4, "S4")
rows = []
for (lab, sd), r in R4.items():
    L = M.losses(r)
    rows.append(dict(controller=lab, seed=sd, **M.torque_ripple(r, 0.02, 0.06),
                     THD_i=M.thd(r, 0.02, 0.06), f_sw_Hz=M.fsw(r), P_sw_W=L["P_sw"]))
d4 = pd.DataFrame(rows)
d4.to_csv(os.path.join(TAB, "S4_ripple_all_seeds.csv"), index=False)
cols4 = ["T_mean", "T_pp", "T_rms", "THD_i", "f_sw_Hz", "P_sw_W"]
a4, m4 = agg(d4, "controller", cols4)
a4.to_csv(os.path.join(TAB, "Table8_S4_torque_ripple.csv"), index=False)
P.torque_zoom([R4[(l, SEEDS[0])] for l, *_ in CTRLS], [l for l, *_ in CTRLS], FIG, "Fig6_S4_torque_ripple", t0=0.05)

# NN weights actually produced online (what does the network do?)
r = R1[("AI-FCS-MPCC (paper)", SEEDS[0])]
import matplotlib  # noqa: E402
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
fig, ax = plt.subplots(2, 1, figsize=(3.5, 2.6), sharex=True)
ax[0].plot(r["log"][:, core.LT] * 1e3, r["log"][:, core.LLD], label=r"$\lambda_d$")
ax[0].plot(r["log"][:, core.LT] * 1e3, r["log"][:, core.LLQ], label=r"$\lambda_q$")
ax[0].set_ylabel("weight")
ax[0].legend(fontsize=6)
ax[1].plot(r["log"][:, core.LT] * 1e3, r["log"][:, core.LLD] / r["log"][:, core.LLQ], "k")
ax[1].set_ylabel(r"$\lambda_d/\lambda_q$")
ax[1].set_xlabel("Time (ms)")
fig.tight_layout()
P.save(fig, FIG, "FigA_paper_NN_weights_S1")

# ------------------------------------------------------------------ 3. robustness (Table 9) + Monte Carlo
print("[3] robustness cases on S2")
cases = [("Nominal", {}), ("Rs +50%", {"Rs": 1.5}), ("Ld,Lq +20%", {"Ld": 1.2, "Lq": 1.2}),
         ("Ld,Lq -20%", {"Ld": 0.8, "Lq": 0.8}), ("psi_f -10%", {"psi": 0.9}),
         ("J +30% (mass)", {"J": 1.3}), ("Vdc -15%", {"Vdc": 0.85})]
sc2s = S.grade_step(cfg, TS, T=0.3)
rows = []
for cname, mm in cases:
    Rr = run_set(sc2s, "S5_" + cname.replace(" ", "").replace("%", "pct").replace(",", ""),
                 seeds=SEEDS[:3], mismatch=mm, keep_seed0=False)
    for (lab, sd), r in Rr.items():
        rows.append(dict(case=cname, controller=lab, seed=sd, **M.tracking(r, 0.05, 0.3),
                         **M.torque_ripple(r, 0.2, 0.3), ok=r["ok"]))
d5 = pd.DataFrame(rows)
d5.to_csv(os.path.join(TAB, "S5_robustness_all_seeds.csv"), index=False)
piv = d5.groupby(["case", "controller"], sort=False)["RMSE_iq"].mean().unstack()[[l for l, *_ in CTRLS]]
piv = piv.reindex([c for c, _ in cases])
piv.round(2).to_csv(os.path.join(TAB, "Table9_robustness_RMSE_iq.csv"))
pivT = d5.groupby(["case", "controller"], sort=False)["T_rms"].mean().unstack()[[l for l, *_ in CTRLS]]
pivT.reindex([c for c, _ in cases]).round(2).to_csv(os.path.join(TAB, "Table9b_robustness_T_rms.csv"))

print(f"[3] Monte Carlo ({N_MC} plant samples)")
rng = np.random.default_rng(12345)
rows = []
for i in range(N_MC):
    mm = dict(Rs=rng.uniform(1.0, 1.5), Ld=rng.uniform(0.8, 1.2), Lq=rng.uniform(0.8, 1.2),
              psi=rng.uniform(0.9, 1.0), J=rng.uniform(1.0, 1.3), Vdc=rng.uniform(0.85, 1.0))
    for lab, cid, ov, nn in CTRLS:
        r = runner.run(cid, cfg, sc2s, nn=nn, mismatch=mm, seed=100 + i, ctrl_overrides=ov)
        record(r, "MC")
        rows.append(dict(sample=i, controller=lab, ok=r["ok"], **{f"k_{k}": v for k, v in mm.items()},
                         **M.tracking(r, 0.05, 0.3), **M.torque_ripple(r, 0.2, 0.3),
                         **M.speed_metrics(r, 0.05)))
d6 = pd.DataFrame(rows)
d6.to_csv(os.path.join(TAB, "MonteCarlo_all_samples.csv"), index=False)
mc = d6.groupby("controller", sort=False).agg(
    RMSE_iq_mean=("RMSE_iq", "mean"), RMSE_iq_std=("RMSE_iq", "std"), RMSE_iq_p95=("RMSE_iq", lambda x: np.percentile(x, 95)),
    T_rms_mean=("T_rms", "mean"), T_rms_std=("T_rms", "std"), speed_dev_mean=("speed_dev_rpm", "mean"),
    failed_runs=("ok", lambda x: int((~x.astype(bool)).sum())))
# paired comparison vs FCS (1,1): fraction of samples where controller is better
base = d6[d6.controller == "FCS-MPCC (1,1)"].set_index("sample")["RMSE_iq"]
mc["frac_better_than_FCS11"] = [float((d6[d6.controller == l].set_index("sample")["RMSE_iq"] < base).mean()) for l in mc.index]
mc.round(3).to_csv(os.path.join(TAB, "Table10_MonteCarlo.csv"))

# ------------------------------------------------------------------ 4. switching-frequency-matched comparison
print("[4] ripple vs switching frequency (1500 rpm, 200 N m)")
sc4b = S.steady(cfg, TS, 1500.0, 200.0, T=0.04)
rows = []
for lab, cid, grid in [("PI-SVPWM", core.PI, [200e-6, 100e-6, 50e-6, 25e-6]),
                       ("FCS-MPCC (1,1)", core.FCS, [50e-6, 25e-6, 20e-6, 12.5e-6, 10e-6, 7.5e-6])]:
    for Tsx in grid:
        for sd in SEEDS[:3]:
            r = runner.run(cid, cfg, sc4b, seed=sd, Ts=Tsx, ctrl_overrides={"nsub": int(round(Tsx / 0.25e-6))})
            record(r, "fsw_sweep")
            L = M.losses(r)
            rows.append(dict(controller=lab, Ts_us=Tsx * 1e6, seed=sd, f_sw_Hz=M.fsw(r),
                             **M.torque_ripple(r, 0.01, 0.04), THD_i=M.thd(r, 0.01, 0.04), P_sw_W=L["P_sw"],
                             **IL.run_columns(r, cfg, IP, 0.01, 0.04)))
d7 = pd.DataFrame(rows)
d7.to_csv(os.path.join(TAB, "fsw_sweep_all_seeds.csv"), index=False)
g7 = d7.groupby(["controller", "Ts_us"], sort=False).mean(numeric_only=True).reset_index()
# Table 11 keeps its original columns; the stator iron-loss columns [W] are appended (P_fe_* = nominal model)
t11cols = ["controller", "Ts_us", "seed", "f_sw_Hz", "T_mean", "T_pp", "T_rms", "THD_i", "P_sw_W",
           "P_fe", "P_fe_fund", "P_fe_harm", "P_fe_h_harm", "P_fe_c_harm", "P_fe_e_harm"]
g7[t11cols].round(3).to_csv(os.path.join(TAB, "Table11_fsw_matched.csv"), index=False)
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.3))
for lab, mk in [("PI-SVPWM", "o-"), ("FCS-MPCC (1,1)", "s-")]:
    g = g7[g7.controller == lab]
    ax[0].plot(g.f_sw_Hz / 1e3, g.T_rms, mk, color=P.COL[lab], label=lab, ms=3)
    ax[1].plot(g.f_sw_Hz / 1e3, 100 * g.THD_i, mk, color=P.COL[lab], label=lab, ms=3)
ax[0].set_xlabel("Average device switching frequency (kHz)")
ax[1].set_xlabel("Average device switching frequency (kHz)")
ax[0].set_ylabel(r"Torque ripple RMS (N m)")
ax[1].set_ylabel(r"Phase-current THD (%)")
ax[0].legend(fontsize=6)
fig.tight_layout()
P.save(fig, FIG, "Fig8_ripple_vs_fsw")

# ------------------------------------------------------------------ 4b. total drive loss (Cu + inverter + stator iron) vs switching frequency
print("[4b] total drive loss vs f_sw at 1500 rpm / 200 N m and 3000 rpm / 30 N m (iron loss included)")
FSW_POINTS = [(1500.0, 200.0), (3000.0, 30.0)]
FSW_FAMILIES = [("PI-SVPWM (carrier sweep)", core.PI, {}, [("Ts", t) for t in (800e-6, 600e-6, 400e-6, 275e-6, 200e-6, 100e-6, 50e-6, 25e-6)]),
                ("FCS-MPCC lam_sw=0 (Ts sweep)", core.FCS, {"lam_sw": 0.0}, [("Ts", t) for t in (50e-6, 25e-6, 20e-6, 12.5e-6, 10e-6, 7.5e-6, 5e-6)]),
                ("FCS-MPCC Ts=10us (lam_sw sweep)", core.FCS, {}, [("lam", l) for l in cfg["ext_sw"]["lam_sw_grid"]])]
rows = []
for rpm_p, tq_p in FSW_POINTS:
    for fam, cid, ov0, grid in FSW_FAMILIES:
        for kind, val in grid:
            Tsx = val if kind == "Ts" else TS_FAST
            ov = dict(ov0, nsub=int(round(Tsx / 0.25e-6)))
            if kind == "lam":
                ov["lam_sw"] = val
            sc = S.steady(cfg, Tsx, rpm_p, tq_p, T=0.04)
            for sd in SEEDS[:3]:
                r = runner.run(cid, cfg, sc, seed=sd, Ts=Tsx, ctrl_overrides=ov)
                record(r, "S4b_fsw_total_loss")
                L = M.losses(r)
                rows.append(dict(rpm=rpm_p, torque=tq_p, family=fam, Ts_us=Tsx * 1e6, lam_sw=ov.get("lam_sw", np.nan), seed=sd,
                                 f_sw_Hz=M.fsw(r), T_rms=M.torque_ripple(r, 0.01, 0.04)["T_rms"],
                                 P_cu=L["P_cu"], P_cond=L["P_cond"], P_sw=L["P_sw"], P_mech=L["P_mech"],
                                 **IL.run_columns(r, cfg, IP, 0.01, 0.04)))
d4b = pd.DataFrame(rows)
d4b["P_noiron"] = d4b.P_cu + d4b.P_cond + d4b.P_sw
d4b["P_total"] = d4b.P_noiron + d4b.P_fe
d4b.to_csv(os.path.join(TAB, "fsw_total_loss_all_seeds.csv"), index=False)
keep = ["f_sw_Hz", "T_rms", "P_cu", "P_cond", "P_sw", "P_noiron", "P_fe", "P_fe_fund", "P_fe_harm", "P_fe_h_harm", "P_fe_c_harm",
        "P_fe_e_harm", "P_total", "ws_fft_resolved"]
g4b = d4b.groupby(["rpm", "torque", "family", "Ts_us", "lam_sw"], sort=False, dropna=False)[keep].mean().reset_index()
g4b.round(3).to_csv(os.path.join(TAB, "Table11b_fsw_total_loss.csv"), index=False)
opt = []
for (rpm_p, tq_p), g in g4b.groupby(["rpm", "torque"], sort=False):
    for fam, gg in g.groupby("family", sort=False):
        a, b = gg.loc[gg.P_noiron.idxmin()], gg.loc[gg.P_total.idxmin()]
        Tlim_ = cfg["ext_sw"]["ripple_limit_pct_of_peak"] / 100 * 430.0     # same 3 % of peak torque ripple limit as the lam_sw study
        gc = gg[gg.T_rms <= Tlim_]
        c = gc.loc[gc.P_total.idxmin()] if len(gc) else b
        opt.append(dict(rpm=rpm_p, torque=tq_p, family=fam, fsw_opt_noiron_kHz=a.f_sw_Hz / 1e3, P_noiron_min_W=a.P_noiron,
                        fsw_opt_with_iron_kHz=b.f_sw_Hz / 1e3, P_total_min_W=b.P_total, P_fe_at_opt_W=b.P_fe,
                        Trms_at_opt_Nm=b.T_rms, P_fe_harm_at_opt_W=b.P_fe_harm,
                        fsw_opt_with_iron_ripple_limited_kHz=c.f_sw_Hz / 1e3, P_total_ripple_limited_W=c.P_total,
                        Trms_ripple_limited_Nm=c.T_rms, ripple_limit_Nm=Tlim_,
                        fsw_range_kHz=f"{gg.f_sw_Hz.min() / 1e3:.1f}-{gg.f_sw_Hz.max() / 1e3:.1f}",
                        optimum_at_sweep_boundary=bool(b.f_sw_Hz in (gg.f_sw_Hz.min(), gg.f_sw_Hz.max()))))
pd.DataFrame(opt).round(3).to_csv(os.path.join(TAB, "Table11c_loss_optimal_fsw.csv"), index=False)
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
FAMCOL = {FSW_FAMILIES[0][0]: "#1f77b4", FSW_FAMILIES[1][0]: "#ff7f0e", FSW_FAMILIES[2][0]: "#d62728"}
for ax, (rpm_p, tq_p) in zip(axs, FSW_POINTS):
    g = g4b[(g4b.rpm == rpm_p) & (g4b.torque == tq_p)]
    for fam, gg in g.groupby("family", sort=False):
        gg = gg.sort_values("f_sw_Hz")
        ax.plot(gg.f_sw_Hz / 1e3, gg.P_total, "o-", color=FAMCOL[fam], ms=3, label=fam)
        ax.plot(gg.f_sw_Hz / 1e3, gg.P_noiron, "--", color=FAMCOL[fam], lw=0.7)
        i = gg.P_total.idxmin()
        ax.plot(gg.f_sw_Hz[i] / 1e3, gg.P_total[i], "k*", ms=7)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 40])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Average device switching frequency (kHz)")
    ax.set_title(f"{rpm_p:.0f} rpm, {tq_p:.0f} N m (solid: with iron, dashed: Cu + inverter only)", fontsize=6)
axs[0].set_ylabel("Drive loss (W)")
axs[0].legend(fontsize=5.5)
fig.tight_layout()
P.save(fig, FIG, "Fig11_total_loss_vs_fsw")

# ------------------------------------------------------------------ 5. reformulation: lam_sw scheduling @ Ts = 10 us
def sw_ctrls(nn_sw, lut_sw, best_sw):
    return [("PI-SVPWM 10 kHz", core.PI, {}, None, None, 100e-6),
            ("PI-SVPWM 3.6 kHz", core.PI, {}, None, None, 275e-6),
            ("FCS-MPCC lam_sw=0", core.FCS, {"lam_sw": 0.0}, None, None, TS_FAST),
            ("FCS-MPCC best fixed lam_sw", core.FCS, {"lam_sw": best_sw}, None, None, TS_FAST),
            ("FCS-MPCC + LUT lam_sw", core.FCS_SWLUT, {}, None, lut_sw, TS_FAST),
            ("FCS-MPCC + NN lam_sw", core.FCS_SWNN, {}, nn_sw, None, TS_FAST)]


CT_SW = sw_ctrls(nn_sw, lut_sw, best_sw)
for rule in ["abs", "rel"]:
    if rule == "abs":
        nn_r, lut_r, best_r, rep_r = nn_sw, lut_sw, best_sw, rep_sw
    else:
        nn_r, lut_r, best_r, rep_r = training.train_sw_scheduler(cfg, TS_FAST, os.path.join(RES, "training_sw_Ts10us_rel"), rule="rel")
    print(f"[5] lam_sw scheduling ({rule} ripple rule) on held-out operating points; best fixed = {best_r}")
    pct = cfg["ext_sw"]["ripple_limit_pct_of_peak"] / 100
    rows = []
    for sp in [750.0, 2250.0, 5250.0, 7000.0]:
        for tq in [75.0, 250.0, 350.0]:
            sc = S.steady(cfg, TS_FAST, sp, tq, T=0.03)
            if sc is None:
                continue
            lim = pct * 430.0 if rule == "abs" else pct * tq
            for lab, cid, ov, nn, lut, Tsx in sw_ctrls(nn_r, lut_r, best_r):
                for sd in SEEDS[:3]:
                    ov2 = dict(ov, nsub=int(round(Tsx / 0.25e-6)))
                    r = runner.run(cid, cfg, sc, nn=nn, lut=lut, seed=sd, Ts=Tsx, ctrl_overrides=ov2)
                    record(r, f"S6_sw_heldout_{rule}")
                    L = M.losses(r)
                    tr = M.torque_ripple(r, 0.01, 0.03)
                    rows.append(dict(rule=rule, rpm=sp, torque=tq, controller=lab, seed=sd, f_sw_Hz=M.fsw(r),
                                     T_rms=tr["T_rms"], P_sw=L["P_sw"], P_cond=L["P_cond"], P_cu=L["P_cu"],
                                     P_loss=L["P_sw"] + L["P_cond"] + L["P_cu"], P_mech=L["P_mech"],
                                     lam_sw_used=float(np.median(r["log"][:, core.LLSW])),
                                     ripple_ok=tr["T_rms"] <= lim))
    d8 = pd.DataFrame(rows)
    d8.to_csv(os.path.join(TAB, f"S6_lam_sw_heldout_{rule}_all.csv"), index=False)
    t8 = d8.groupby("controller", sort=False).agg(P_loss_W=("P_loss", "mean"), P_sw_W=("P_sw", "mean"),
                                                   T_rms_Nm=("T_rms", "mean"), T_rms_max=("T_rms", "max"),
                                                   f_sw_Hz=("f_sw_Hz", "mean"), lam_sw_median=("lam_sw_used", "median"),
                                                   ripple_ok_frac=("ripple_ok", "mean"))
    t8.round(2).to_csv(os.path.join(TAB, f"Table12_lam_sw_heldout_{rule}.csv"))

print("[5] S3 speed manoeuvre with lam_sw controllers (abs rule; energy over manoeuvre)")
sc3f = S.speed_step(cfg, TS_FAST, T=1.0)
rows = []
for lab, cid, ov, nn, lut, Tsx in CT_SW:
    for sd in SEEDS[:3]:
        ov2 = dict(ov, nsub=int(round(Tsx / 0.25e-6)))
        r = runner.run(cid, cfg, sc3f, nn=nn, lut=lut, seed=sd, Ts=Tsx, ctrl_overrides=ov2)
        record(r, "S7_sw_manoeuvre")
        E = r["E"]
        rows.append(dict(controller=lab, seed=sd, E_sw_J=E[2], E_cond_J=E[1], E_cu_J=E[0],
                         E_loss_J=E[0] + E[1] + E[2], E_mech_J=E[3], f_sw_Hz=M.fsw(r),
                         t_settle_ms=M.speed_metrics(r, sc3f["t_event"])["t_recover_ms"],
                         **M.torque_ripple(r, 0.9, 1.0)))
d9 = pd.DataFrame(rows)
d9.to_csv(os.path.join(TAB, "S7_lam_sw_manoeuvre_all.csv"), index=False)
d9.groupby("controller", sort=False).mean(numeric_only=True).drop(columns="seed").round(2).to_csv(
    os.path.join(TAB, "Table13_lam_sw_manoeuvre.csv"))

# ------------------------------------------------------------------ 6. simulated loss maps, efficiency maps, WLTC energy, loss sensitivity
from aifcs import wltc  # noqa: E402
print("[6] loss maps over the torque-speed envelope (MTPA + field weakening)")
MAP_RPM = [200.0, 500.0, 1000.0, 2000.0, 3000.0, 4500.0, 6000.0, 7500.0, 9500.0]
MAP_T = [5.0, 15.0, 30.0, 60.0, 100.0, 150.0, 200.0, 300.0, 430.0]
rows = []
for lab, cid, ov, nn, lut, Tsx in CT_SW:
    for sp in MAP_RPM:
        for tq in MAP_T:
            sc = S.steady(cfg, Tsx, sp, tq, T=0.03)
            if sc is None:
                continue
            r = runner.run(cid, cfg, sc, nn=nn, lut=lut, seed=7, Ts=Tsx,
                           ctrl_overrides=dict(ov, nsub=int(round(Tsx / 0.25e-6))))
            record(r, "S8_lossmap")
            L = M.losses(r)
            tr = M.torque_ripple(r, 0.01, 0.03)
            rows.append(dict(controller=lab, rpm=sp, torque=tq, P_cu=L["P_cu"], P_cond=L["P_cond"], P_sw=L["P_sw"],
                             P_mech=L["P_mech"], T_rms=tr["T_rms"], f_sw_Hz=M.fsw(r), ok=r["ok"],
                             **IL.run_columns(r, cfg, IP, 0.01, 0.03)))
dm = pd.DataFrame(rows)
dm.to_csv(os.path.join(TAB, "lossmap_all_controllers.csv"), index=False)


def grid_of(df, lab, k_sw=1.0, k_cond=1.0, col=None):
    g = np.full((len(MAP_RPM), len(MAP_T)), np.nan)
    sub = df[df.controller == lab]
    for _, x in sub.iterrows():
        i, j = MAP_RPM.index(x.rpm), MAP_T.index(x.torque)
        g[i, j] = x[col] if col else x.P_cu + k_cond * x.P_cond + k_sw * x.P_sw
    return g


def grid_fn(df, lab, fn):
    """Map grid of fn(row) for one controller."""
    g = np.full((len(MAP_RPM), len(MAP_T)), np.nan)
    for _, x in df[df.controller == lab].iterrows():
        g[MAP_RPM.index(x.rpm), MAP_T.index(x.torque)] = fn(x)
    return g


fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.6), sharey=True, sharex=True)
lev = np.linspace(0.60, 1.0, 21)
env = [min(430.0, params.max_torque(cfg, r)) for r in np.linspace(200, 9500, 60)]
for row_i, (with_fe, rowlab) in enumerate([(False, "Cu + inv."), (True, "Cu + inv. + stator iron")]):
    for ax, lab in zip(axs[row_i], ["PI-SVPWM 10 kHz", "FCS-MPCC + NN lam_sw"]):
        Pl = grid_of(dm, lab) + (grid_fn(dm, lab, lambda x: x.P_fe) if with_fe else 0.0)
        Pm = grid_of(dm, lab, col="P_mech")
        eta = Pm / (Pm + Pl)
        pd.DataFrame(eta, index=MAP_RPM, columns=MAP_T).round(4).to_csv(
            os.path.join(TAB, f"effmap_{lab.replace(' ', '_')}{'_withiron' if with_fe else ''}.csv"))
        cs = ax.contourf(MAP_RPM, MAP_T, np.ma.masked_invalid(eta).T, levels=lev, cmap="viridis", extend="min")
        ax.contour(MAP_RPM, MAP_T, np.ma.masked_invalid(eta).T, levels=[0.8, 0.9, 0.95], colors="w", linewidths=0.5)
        ax.plot(np.linspace(200, 9500, 60), env, "k-", lw=0.8)
        ax.set_title(f"{lab.replace('FCS-MPCC + NN lam_sw', 'FCS + NN lam_sw')}: {rowlab}", fontsize=6.5)
    axs[row_i, 0].set_ylabel("Torque (N m)")
for ax in axs[1]:
    ax.set_xlabel("Speed (rpm)")
fig.colorbar(cs, ax=axs, label="Efficiency (motoring, shaft / (shaft + losses))")
P.save(fig, FIG, "Fig7_efficiency_maps_simulated")

print("[6] WLTC Class 3b quasi-static energy + inverter-loss sensitivity")
dem = wltc.demand(cfg)
fig, ax = plt.subplots(2, 1, figsize=(3.5, 2.6), sharex=True)
ax[0].plot(dem["v"] * 3.6, "k", lw=0.5)
ax[0].set_ylabel("km/h")
ax[1].plot(dem["T"], "k", lw=0.4)
ax[1].set_ylabel("Motor torque (N m)")
ax[1].set_xlabel("Time (s)")
fig.tight_layout()
P.save(fig, FIG, "FigB_WLTC_demand")
rows = []
for k_sw, k_cond, tag in [(1, 1, "nominal"), (0.5, 1, "Esw x0.5"), (2, 1, "Esw x2"), (1, 0.5, "r_on x0.5"), (1, 2, "r_on x2")]:
    for lab, *_ in CT_SW:
        e = wltc.energy(cfg, dict(rpm=np.array(MAP_RPM), torque=np.array(MAP_T), P=grid_of(dm, lab, k_sw, k_cond)), dem)
        g = dict(rpm=np.array(MAP_RPM), torque=np.array(MAP_T), P=grid_of(dm, lab, col="T_rms"))
        rows.append(dict(loss_case=tag, controller=lab, **e,
                         T_rms_mean_Nm=wltc.time_mean(cfg, g, dem),
                         T_rms_rel_mean=wltc.time_mean(cfg, g, dem, relative=True)))
dw = pd.DataFrame(rows)
dw.round(3).to_csv(os.path.join(TAB, "Table14_WLTC_energy_and_sensitivity.csv"), index=False)
dw.pivot(index="controller", columns="loss_case", values="loss_Wh_per_km").round(2)[
    ["nominal", "Esw x0.5", "Esw x2", "r_on x0.5", "r_on x2"]].to_csv(os.path.join(TAB, "Table14b_WLTC_loss_Wh_per_km.csv"))
# best fixed lam_sw re-selected for each loss assumption (labels depend on the loss model)
gs = pd.read_csv(os.path.join(RES, "training_sw_Ts10us", "sw_grid_search.csv"))
sel = []
for k_sw, k_cond, tag in [(1, 1, "nominal"), (0.5, 1, "Esw x0.5"), (2, 1, "Esw x2"), (1, 0.5, "r_on x0.5"), (1, 2, "r_on x2")]:
    gs["P_tot"] = gs.P_cu + k_cond * gs.P_cond + k_sw * gs.P_sw
    Tlim = rep_sw["T_limit_Nm"]
    feas = gs.groupby("lam_sw").apply(lambda g: bool((g.T_rms <= Tlim).all()), include_groups=False)
    tot = gs.groupby("lam_sw").P_tot.sum()
    best = tot[feas[feas].index].idxmin() if feas.any() else 0.0
    sel.append(dict(loss_case=tag, best_fixed_lam_sw=best))
pd.DataFrame(sel).to_csv(os.path.join(TAB, "Table14c_best_lam_sw_vs_loss_model.csv"), index=False)

# ------------------------------------------------------------------ 6a. WLTC with stator iron loss (+ coefficient sensitivity)
print("[6a] WLTC with stator iron loss: nominal + coefficient sensitivity + break-even harmonic multiplier")
pd.DataFrame([dict(item=k, value=v) for k, v in IP.info.items()]
             + [dict(item=k, value=getattr(IP, k)) for k in ("k_h", "k_c", "k_e", "alpha", "build_factor", "thickness", "sigma", "mu_r_eff", "m_t", "m_y", "K_t", "K_y")]
             ).to_csv(os.path.join(TAB, "iron_model_parameters.csv"), index=False)
RG, TG = np.array(MAP_RPM), np.array(MAP_T)
IRON_CASES = [("nominal", IP), ("k_h x0.5", IP.scaled(k_h=0.5)), ("k_h x2", IP.scaled(k_h=2)),
              ("k_c x0.5", IP.scaled(k_c=0.5)), ("k_c x2", IP.scaled(k_c=2)),
              ("k_e x0.5", IP.scaled(k_e=0.5)), ("k_e x2", IP.scaled(k_e=2)),
              ("build factor x0.5", IP.scaled(build=0.5)), ("build factor x2", IP.scaled(build=2)),
              ("no skin screening (upper bound)", IP.scaled(screening=False)),
              ("FFT-resolved band only (lower bound)", IP.scaled(resolved_only=True)),
              ("harmonic part x0.25", IP.scaled(harm=0.25)), ("harmonic part x0.5", IP.scaled(harm=0.5)),
              ("harmonic part x2", IP.scaled(harm=2)), ("harmonic part x3", IP.scaled(harm=3)),
              ("fundamental only (harmonic x0)", IP.scaled(harm=0.0)),
              ("adverse: k_c x2, k_e x2, no screening, harm x2", IP.scaled(k_c=2, k_e=2, screening=False, harm=2))]
CTL = [lab for lab, *_ in CT_SW]


def fe_grid(lab, ipc, key="P_fe"):
    return grid_fn(dm, lab, lambda x: IL.combine_row(x, ipc)[key])


def wltc_loss(lab, ipc, with_iron=True, key="P_fe"):
    P0 = grid_of(dm, lab)
    Pg = (P0 + fe_grid(lab, ipc, key)) if with_iron else P0
    return wltc.energy(cfg, dict(rpm=RG, torque=TG, P=Pg), dem)


rows = []
for cname, ipc in IRON_CASES:
    for lab in CTL:
        e0 = wltc_loss(lab, ipc, False)
        e1 = wltc_loss(lab, ipc, True)
        fe_f = wltc.energy(cfg, dict(rpm=RG, torque=TG, P=fe_grid(lab, ipc, "P_fund")), dem)["loss_Wh_per_km"]
        fe_h = wltc.energy(cfg, dict(rpm=RG, torque=TG, P=fe_grid(lab, ipc, "P_harm")), dem)["loss_Wh_per_km"]
        rows.append(dict(iron_case=cname, controller=lab, loss_noiron_Wh_per_km=e0["loss_Wh_per_km"],
                         loss_with_iron_Wh_per_km=e1["loss_Wh_per_km"], iron_Wh_per_km=fe_f + fe_h,
                         iron_fund_Wh_per_km=fe_f, iron_harm_Wh_per_km=fe_h,
                         iron_share_of_drive_loss=(fe_f + fe_h) / e1["loss_Wh_per_km"],
                         harm_share_of_iron=fe_h / (fe_f + fe_h) if fe_f + fe_h > 0 else 0.0,
                         battery_Wh_per_km_with_iron=e1["Wh_per_km"], battery_Wh_per_km_noiron=e0["Wh_per_km"]))
dwi = pd.DataFrame(rows)
dwi["rank_noiron"] = dwi.groupby("iron_case").loss_noiron_Wh_per_km.rank(method="min").astype(int)
dwi["rank_with_iron"] = dwi.groupby("iron_case").loss_with_iron_Wh_per_km.rank(method="min").astype(int)
dwi.round(4).to_csv(os.path.join(TAB, "Table14e_WLTC_iron_sensitivity.csv"), index=False)
nom = dwi[dwi.iron_case == "nominal"].set_index("controller")
nom[["loss_noiron_Wh_per_km", "loss_with_iron_Wh_per_km", "iron_Wh_per_km", "iron_share_of_drive_loss",
     "iron_fund_Wh_per_km", "iron_harm_Wh_per_km", "harm_share_of_iron", "rank_noiron", "rank_with_iron",
     "battery_Wh_per_km_noiron", "battery_Wh_per_km_with_iron"]].round(3).to_csv(os.path.join(TAB, "Table14d_WLTC_with_iron_loss.csv"))
# ranking flips per case among the four reference controllers
REF4 = ["PI-SVPWM 10 kHz", "PI-SVPWM 3.6 kHz", "FCS-MPCC lam_sw=0", "FCS-MPCC best fixed lam_sw"]
flip = []
for cname, g in dwi[dwi.controller.isin(REF4)].groupby("iron_case", sort=False):
    g = g.set_index("controller")
    o0 = list(g.loss_noiron_Wh_per_km.sort_values().index)
    o1 = list(g.loss_with_iron_Wh_per_km.sort_values().index)
    flip.append(dict(iron_case=cname, order_noiron=" < ".join(o0), order_with_iron=" < ".join(o1), ranking_changed=o0 != o1,
                     best_with_iron=o1[0],
                     FCS_vs_PI3p6k_pct=100 * (g.loc["FCS-MPCC best fixed lam_sw", "loss_with_iron_Wh_per_km"] / g.loc["PI-SVPWM 3.6 kHz", "loss_with_iron_Wh_per_km"] - 1),
                     FCS_vs_PI10k_pct=100 * (g.loc["FCS-MPCC best fixed lam_sw", "loss_with_iron_Wh_per_km"] / g.loc["PI-SVPWM 10 kHz", "loss_with_iron_Wh_per_km"] - 1),
                     FCS_vs_PI3p6k_noiron_pct=100 * (g.loc["FCS-MPCC best fixed lam_sw", "loss_noiron_Wh_per_km"] / g.loc["PI-SVPWM 3.6 kHz", "loss_noiron_Wh_per_km"] - 1)))
pd.DataFrame(flip).round(2).to_csv(os.path.join(TAB, "Table14f_WLTC_ranking_vs_iron_case.csv"), index=False)
# break-even multiplier m* on the harmonic (PWM) iron-loss part: loss(m) = A + F + m H is exactly linear in m
be = []
for lab_ref in ["PI-SVPWM 3.6 kHz", "PI-SVPWM 10 kHz", "FCS-MPCC lam_sw=0"]:
    def lin(lab):
        e0 = wltc_loss(lab, IP, False)["loss_Wh_per_km"]
        fu = wltc.energy(cfg, dict(rpm=RG, torque=TG, P=fe_grid(lab, IP, "P_fund")), dem)["loss_Wh_per_km"]
        ha = wltc.energy(cfg, dict(rpm=RG, torque=TG, P=fe_grid(lab, IP, "P_harm")), dem)["loss_Wh_per_km"]
        return e0 + fu, ha
    a_f, h_f = lin("FCS-MPCC best fixed lam_sw")
    a_r, h_r = lin(lab_ref)
    mstar = (a_r - a_f) / (h_f - h_r) if h_f > h_r else np.inf
    be.append(dict(reference=lab_ref, fcs_noharm_Wh_per_km=a_f, ref_noharm_Wh_per_km=a_r, fcs_harm_nominal_Wh_per_km=h_f,
                   ref_harm_nominal_Wh_per_km=h_r, breakeven_harmonic_multiplier=mstar,
                   note="FCS (best fixed lam_sw) loses to the reference only if the harmonic iron part is multiplied by more than this"))
pd.DataFrame(be).round(3).to_csv(os.path.join(TAB, "Table14g_breakeven_harmonic_multiplier.csv"), index=False)
# lam_sw selection with iron loss (grid search of the scheduler training; labels of the NN/LUT themselves are NOT retrained)
sel2 = []
for tag, ipc in [("no iron", None), ("iron nominal", IP), ("iron, harmonic x3", IP.scaled(harm=3)), ("iron, no screening", IP.scaled(screening=False))]:
    gs["P_tot"] = gs.P_cu + gs.P_cond + gs.P_sw
    if ipc is not None:
        gs["P_tot"] += [IL.combine(dict(**{k: float(x["ws_" + k]) for k in IL.STAT_COLS}), ipc)["P_fe"] for _, x in gs.iterrows()]
    feas = gs.groupby("lam_sw").apply(lambda g: bool((g.T_rms <= rep_sw["T_limit_Nm"]).all()), include_groups=False)
    tot = gs.groupby("lam_sw").P_tot.sum()
    sel2.append(dict(loss_model=tag, best_fixed_lam_sw=tot[feas[feas].index].idxmin(),
                     sum_P_by_lam_sw={str(k): round(float(v), 0) for k, v in tot.items()}))
pd.DataFrame(sel2).to_csv(os.path.join(TAB, "Table14h_best_lam_sw_with_iron.csv"), index=False)

# Fig 10: WLTC drive-loss breakdown (stacked) for the reference controllers
fig, ax = plt.subplots(figsize=(3.5, 2.6))
labs = ["PI-SVPWM 10 kHz", "PI-SVPWM 3.6 kHz", "FCS-MPCC lam_sw=0", "FCS-MPCC best fixed lam_sw"]
short = ["PI 10 kHz", "PI 3.6 kHz", r"FCS $\lambda_{sw}$=0", r"FCS $\lambda_{sw}$=1000"]
bot = np.zeros(len(labs))
for key, col, nm in [("loss_noiron_Wh_per_km", "#4c72b0", "Cu + inverter"), ("iron_fund_Wh_per_km", "#55a868", "iron, fundamental"),
                     ("iron_harm_Wh_per_km", "#c44e52", "iron, PWM harmonics")]:
    v = np.array([nom.loc[l, key] for l in labs])
    ax.bar(short, v, bottom=bot, color=col, label=nm)
    bot += v
ax.set_ylabel("WLTC 3b drive loss (Wh/km)")
ax.set_ylim(0, 16)
ax.legend(fontsize=6, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False)
plt.setp(ax.get_xticklabels(), rotation=15, fontsize=6)
fig.tight_layout()
P.save(fig, FIG, "Fig12_WLTC_loss_breakdown_with_iron")

# ------------------------------------------------------------------ 6b. inductance sensitivity (stand-in for saturation)
print("[6b] inductance sensitivity: plant AND controller model L x {1, 1.5, 2}")
import copy  # noqa: E402
rows = []
for kL in [1.0, 1.5, 2.0]:
    cfgL = copy.deepcopy(cfg)
    cfgL["motor"]["Ld"] *= kL
    cfgL["motor"]["Lq"] *= kL
    for tq in [100.0, 300.0]:
        sc = S.steady(cfgL, TS, 1500.0, tq, T=0.04)
        for lab, cid, Tsx, ov, nn in [("PI-SVPWM", core.PI, 100e-6, {}, None), ("PI-SVPWM", core.PI, 50e-6, {}, None),
                                      ("FCS-MPCC (1,1)", core.FCS, 50e-6, {}, None), ("FCS-MPCC (1,1)", core.FCS, 20e-6, {}, None),
                                      ("FCS-MPCC (1,1)", core.FCS, 10e-6, {}, None),
                                      ("FCS-MPCC best fixed", core.FCS, 50e-6, {"lam_d": best_pair[0], "lam_q": best_pair[1]}, None),
                                      ("AI-FCS-MPCC (paper)", core.FCS_NN, 50e-6, {}, nn_paper)]:
            for sd in SEEDS[:3]:
                r = runner.run(cid, cfgL, sc, nn=nn, seed=sd, Ts=Tsx, ctrl_overrides=dict(ov, nsub=int(round(Tsx / 0.25e-6))))
                record(r, "S9_L_sensitivity")
                tr = M.torque_ripple(r, 0.01, 0.04)
                rows.append(dict(L_factor=kL, torque=tq, controller=lab, Ts_us=Tsx * 1e6, seed=sd,
                                 f_sw_Hz=M.fsw(r), T_rms=tr["T_rms"], THD_i=M.thd(r, 0.01, 0.04)))
dL = pd.DataFrame(rows)
dL.to_csv(os.path.join(TAB, "S9_L_sensitivity_all.csv"), index=False)
dL.groupby(["L_factor", "torque", "controller", "Ts_us"], sort=False).mean(numeric_only=True).drop(columns="seed").round(3).to_csv(
    os.path.join(TAB, "Table15_L_sensitivity.csv"))

# ------------------------------------------------------------------ 6c. M1: variable-switching-frequency PWM / DPWM baselines, ripple-constrained WLTC comparison
sys.path.insert(0, HERE)
import vsf_baselines  # noqa: E402
print("[6c] ripple-constrained comparison: PI-SVPWM / DPWM1 / DPWM-I with scheduled carrier vs FCS-MPCC with scheduled lam_sw")
vsf_baselines.run(cfg, MAP_RPM, MAP_T, SEEDS[:2] if args.quick else SEEDS[:3], TAB, FIG,
                  record=record, quick=args.quick)

# ------------------------------------------------------------------ 7. computational burden (operation count, not timing)
ops = pd.DataFrame([
    dict(controller="PI-SVPWM", flops_est=60, transcendental="sin, cos", note="2 PI + decoupling + Park^-1 + min-max SVPWM"),
    dict(controller="Deadbeat-SVPWM", flops_est=90, transcendental="2x(sin, cos)", note="delay comp. + 1-step inverse model + SVPWM"),
    dict(controller="FCS-MPCC", flops_est=30 + 8 * 28, transcendental="2x(sin, cos)", note="delay comp. + 8 candidates x (rotation, prediction, cost)"),
    dict(controller="AI-FCS-MPCC (paper)", flops_est=30 + 8 * 28 + 2 * (4 * 10 + 10 * 2) + 20, transcendental="2x(sin, cos) + 10 tanh + 2 exp", note="FCS + NN 4-10-2"),
    dict(controller="FCS-MPCC + NN lam_sw", flops_est=30 + 8 * 30 + 2 * (2 * 6 + 6) + 10, transcendental="2x(sin, cos) + 6 tanh + exp + pow", note="FCS + NN 2-6-1 (can run at 1 kHz)"),
])
ops.to_csv(os.path.join(TAB, "Table1_operation_count_estimate.csv"), index=False)

manifest["finished"] = datetime.now().isoformat(timespec="seconds")
manifest["total_wall_s"] = round(time.time() - T0, 1)
manifest["n_runs"] = len(manifest["runs"])
manifest["n_failed"] = sum(not x["ok"] for x in manifest["runs"])
manifest["iron_model"] = dict(info=IP.info, k_h=IP.k_h, k_c=IP.k_c, k_e=IP.k_e, alpha=IP.alpha, build_factor=IP.build_factor,
                            thickness=IP.thickness, sigma=IP.sigma, mu_r_eff=IP.mu_r_eff)
manifest["paper_nn"] = rep_paper
manifest["sw_scheduler"] = {k: v for k, v in rep_sw.items() if k != "lut_log10"}
with open(os.path.join(RES, "run_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=1, default=str)
print(f"done: {manifest['n_runs']} runs, {manifest['n_failed']} failed, {manifest['total_wall_s']} s")
