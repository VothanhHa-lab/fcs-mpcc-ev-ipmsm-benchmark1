"""M1: fair variable-switching-frequency PWM baselines and a ripple-constrained WLTC comparison.

Called from scripts/reproduce.py (section 6c) or standalone:
    python3 scripts/vsf_baselines.py            # full
    python3 scripts/vsf_baselines.py --quick    # fewer seeds

Every number written to results/ comes from simulation runs made here (maps) or from
arithmetic on those maps (WLTC evaluation, scaling of loss components).
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from aifcs import core, params, scenarios as S, runner, metrics as M, vsf, plotting as P  # noqa: E402

T_PEAK = 430.0
LIMITS = {"abs3": ("3 % of 430 N m = 12.9 N m (absolute)", vsf.limit_abs(0.03 * T_PEAK)),
          "abs1": ("1 % of 430 N m = 4.3 N m (absolute)", vsf.limit_abs(0.01 * T_PEAK)),
          "rel5": ("5 % of demanded torque, floor 3 N m (relative)", vsf.limit_rel(5.0, 3.0)),
          "rel5f2": ("5 % of demanded torque, floor 2 N m (relative)", vsf.limit_rel(5.0, 2.0))}
SWEEP_ABS = [1.5, 2.0, 2.5, 3.0, 4.3, 6.0, 8.0, 10.0, 12.9, 16.0, 20.0]
LOSS_CASES = [("nominal", 1.0, 1.0, 0.0), ("Esw x0.5", 0.5, 1.0, 0.0), ("Esw x2", 2.0, 1.0, 0.0),
              ("r_on x0.5", 1.0, 0.5, 0.0), ("r_on x2", 1.0, 2.0, 0.0),
              # with stator iron loss incl. PWM harmonics (ironloss.py, nominal [ASM] coefficients)
              ("nominal + iron", 1.0, 1.0, 1.0), ("Esw x0.5 + iron", 0.5, 1.0, 1.0), ("Esw x2 + iron", 2.0, 1.0, 1.0)]
HO_RPM, HO_T = [750.0, 2250.0, 5250.0, 7000.0], [75.0, 250.0, 350.0]
BASE_NAME = {"PI-SVPWM VSF": "PI-SVPWM", "PI-DPWM1 VSF": "PI-DPWM1", "PI-DPWM-I VSF": "PI-DPWM-I",
             "FCS lam_sw sched (Ts=10us)": "FCS-MPCC (Ts = 10 us)", "FCS (Ts, lam_sw) sched": "FCS-MPCC (Ts, lam_sw)"}


def step_response(r, t_step, band=0.05):
    """iq step response from the fine log: bin-average per control period, 5 % settling, overshoot."""
    f = r["fine"]
    Ts = r["Ts"]
    k = (f[:, core.FT] / Ts + 1e-9).astype(int)
    n = k.max() + 1
    iq = np.bincount(k, weights=f[:, core.FIQ], minlength=n) / np.maximum(np.bincount(k, minlength=n), 1)
    ref = r["log"][:, core.LIQREF]
    k0 = int(round(t_step / Ts))
    final, start = ref[-1], ref[max(k0 - 1, 0)]
    step = final - start
    post = iq[k0 + 1:]
    out = np.where(np.abs(post - final) > band * abs(step))[0]
    ts = (out[-1] + 2) * Ts if out.size else Ts
    os_ = max(0.0, float(np.max(np.sign(step) * (post - final))))
    return 1e3 * ts, 100 * os_ / abs(step)


def run(cfg, rpm_grid, torque_grid, seeds, tab_dir, fig_dir, record=None, quick=False):
    t0 = time.time()
    rg, tg = list(rpm_grid), list(torque_grid)
    settings = vsf.settings_list()
    ids = [(s["family"], s["label"]) for s in settings]
    n_runs0 = None

    # ------------------------------------------------------------------ 1. maps
    print(f"[6c] maps: {len(settings)} settings x {len(rg)}x{len(tg)} nodes x {len(seeds)} seeds")
    if os.environ.get("VSF_REUSE_MAPS") and os.path.exists(os.path.join(tab_dir, "vsf_maps_all_seeds.csv")):
        raw = pd.read_csv(os.path.join(tab_dir, "vsf_maps_all_seeds.csv"))     # development shortcut only
    else:
        raw = vsf.build_maps(cfg, rg, tg, seeds, record=record)
        raw.to_csv(os.path.join(tab_dir, "vsf_maps_all_seeds.csv"), index=False)
    dfm = vsf.average_maps(raw)
    dfm["P_loss"] = vsf.ploss(dfm)
    dfm.round(5).to_csv(os.path.join(tab_dir, "vsf_maps_seed_mean.csv"), index=False)
    n_failed_maps = int((~raw.ok).sum())
    nodes = dfm[["rpm", "torque"]].drop_duplicates()
    print(f"     {len(raw)} map runs, {n_failed_maps} non-finite; {len(nodes)} feasible nodes; "
          f"{int(dfm.track_fail.sum())} (setting, node) pairs fail torque tracking")
    unstable = dfm.groupby(["family", "label"], sort=False).track_fail.sum()
    unstable = unstable[unstable > 0].reset_index().rename(columns={"track_fail": "nodes_failing_torque_tracking"})
    unstable["of_nodes"] = len(nodes)
    unstable.to_csv(os.path.join(tab_dir, "Table16h_settings_failing_torque_tracking.csv"), index=False)

    ev = vsf.WltcEval(cfg, rg, tg)
    # cross-check of the evaluator against wltc.energy for one fixed setting
    ser_nom = vsf.setting_series(ev, dfm, ids)
    g = dfm[(dfm.family == "PI-SVPWM") & (dfm.label == "10 kHz")]
    from aifcs import wltc
    e_ref = wltc.energy(cfg, dict(rpm=np.array(rg), torque=np.array(tg),
                                  P=vsf.to_grid(g, "P_loss", rg, tg)), ev.d)["loss_Wh_per_km"]
    k10 = ids.index(("PI-SVPWM", "10 kHz"))
    e_mine = ev.summarise(ser_nom[0][k10], ser_nom[1][k10], ser_nom[2][k10])["loss_Wh_per_km"]
    assert abs(e_ref - e_mine) < 1e-9 * max(1.0, e_ref), (e_ref, e_mine)

    # ------------------------------------------------------------------ 2. ripple-constrained comparison
    K_IDEAL, K_LUT, K_FIX = "scheduled, ideal (per WLTC sample)", "scheduled, 9x9 LUT (nearest node)", "best single setting"
    rows, lut_rows, fixed_rows = [], [], []
    for case, k_sw, k_cond, k_fe in LOSS_CASES:
        ser = ser_nom if case == "nominal" else vsf.setting_series(ev, dfm, ids, k_sw, k_cond, k_fe)
        for lk, (ldesc, lfn) in LIMITS.items():
            for meth, fams in METHODS_ITEMS:
                allowed = [k for k, (f, _) in enumerate(ids) if f in fams]
                sel = vsf.select(dfm, fams, lfn, k_sw, k_cond, k_fe)
                lut = vsf.make_lut(sel, rg, tg, ids)
                ninf = int((~sel.feasible).sum())
                if ninf and case == "nominal":
                    print(f"     note: {meth}, limit {lk}, {case}: {ninf} map nodes have no feasible setting")
                common = dict(loss_case=case, limit=lk, limit_desc=ldesc, method=BASE_NAME[meth], nodes_infeasible=ninf,
                              nodes=len(sel))
                a, _ = vsf.eval_pointwise(ev, ser, allowed, lfn)
                rows.append(dict(**common, kind=K_IDEAL, setting="continuous schedule", **a))
                a = vsf.eval_lut(ev, ser, lut, lfn)
                rows.append(dict(**common, kind=K_LUT, setting="LUT (per operating point)", **a))
                if case == "nominal":
                    for r_ in sel.itertuples():
                        lut_rows.append(dict(limit=lk, method=BASE_NAME[meth], rpm=r_.rpm, torque=r_.torque,
                                             family=r_.family, setting=r_.label, Ts_us=round(r_.Ts_us, 3),
                                             carrier_kHz=(r_.knob / 1e3 if r_.family.startswith("PI") else np.nan),
                                             lam_sw=(r_.knob if r_.family.startswith("FCS") else np.nan),
                                             limit_Nm=r_.limit, feasible=r_.feasible, T_rms=r_.T_rms,
                                             P_loss_W=r_.P_loss))
                cand = vsf.fixed_feasible(dfm, fams, lfn)
                if not cand:
                    rows.append(dict(**common, kind=K_FIX, setting="none feasible at every node"))
                else:   # among the settings that are feasible at every node: smallest WLTC drive loss
                    k = min((ids.index(c) for c in cand), key=lambda kk: ev.summarise(ser[0][kk], ser[1][kk], ser[2][kk])["loss_Wh_per_km"])
                    a = ev.summarise(ser[0][k], ser[1][k], ser[2][k], lfn)
                    rows.append(dict(**common, kind=K_FIX, setting=f"{ids[k][0]} {ids[k][1]}", **a))
    t16 = pd.DataFrame(rows)
    # relative change vs the PI-SVPWM VSF baseline of the same kind, loss case and limit
    ref = t16[(t16.method == "PI-SVPWM") & (t16.kind != K_FIX)].set_index(["kind", "loss_case", "limit"]).loss_Wh_per_km
    t16["loss_vs_PI_VSF_pct"] = [100 * (r.loss_Wh_per_km / ref[(r.kind, r.loss_case, r.limit)] - 1)
                                 if (pd.notna(r.loss_Wh_per_km) and r.kind != K_FIX) else np.nan for r in t16.itertuples()]
    nom = t16[t16.loss_case == "nominal"].drop(columns="loss_case")
    nom.round(4).to_csv(os.path.join(tab_dir, "Table16_ripple_constrained_WLTC.csv"), index=False)
    t16.round(4).to_csv(os.path.join(tab_dir, "Table16b_ripple_constrained_loss_sensitivity.csv"), index=False)
    piv = t16[t16.kind == K_IDEAL].pivot_table(index=["limit", "method"], columns="loss_case",
                                               values="loss_Wh_per_km", sort=False)[[c for c, *_ in LOSS_CASES]]
    piv.round(3).to_csv(os.path.join(tab_dir, "Table16c_scheduled_loss_Wh_per_km_by_loss_case.csv"))
    pd.DataFrame(lut_rows).round(4).to_csv(os.path.join(tab_dir, "Table16d_schedule_LUTs.csv"), index=False)

    # ------------------------------------------------------------------ 3. every fixed setting on WLTC (Pareto curves)
    P_, T_, F_, B_ = ser_nom
    for k, (fam, lab) in enumerate(ids):
        a = ev.summarise(P_[k], T_[k], F_[k])
        d = dict(family=fam, setting=lab, **a)
        for lk, (_, lfn) in LIMITS.items():
            d[f"viol_time_frac_{lk}"] = ev.summarise(P_[k], T_[k], F_[k], lfn)["viol_time_frac"]
        gg = dfm[(dfm.family == fam) & (dfm.label == lab)]
        d["nodes_failing_torque_tracking"] = int(gg.track_fail.sum())
        fixed_rows.append(d)
    tfix = pd.DataFrame(fixed_rows)
    tfix.round(4).to_csv(os.path.join(tab_dir, "Table16e_WLTC_fixed_settings.csv"), index=False)

    # scheduled frontier: sweep of the absolute ripple limit (nominal loss model, ideal schedule)
    sw = []
    for L in SWEEP_ABS:
        lfn = vsf.limit_abs(L)
        for meth, fams in METHODS_ITEMS:
            allowed = [k for k, (f, _) in enumerate(ids) if f in fams]
            a, _ = vsf.eval_pointwise(ev, ser_nom, allowed, lfn)
            sw.append(dict(limit_Nm=L, method=BASE_NAME[meth], **a))
    tsw = pd.DataFrame(sw)
    tsw.round(4).to_csv(os.path.join(tab_dir, "Table16f_scheduled_frontier_sweep.csv"), index=False)

    # carrier-set ablation: does extending the set to 1-2 kHz change the PWM baselines? (ideal schedule, nominal losses)
    knob_of = {(st["family"], st["label"]): st["knob"] for st in settings}
    arows = []
    for fam in [f for f, _ in vsf.PWM_FAMILIES]:
        for lk in ("abs3", "abs1", "rel5"):
            lfn = LIMITS[lk][1]
            for tag, fmin in [("carriers 1-20 kHz (used)", 0.0), ("carriers 2.5-20 kHz", 2500.0)]:
                allowed = [k for k, i_ in enumerate(ids) if i_[0] == fam and knob_of[i_] >= fmin]
                a, _ = vsf.eval_pointwise(ev, ser_nom, allowed, lfn)
                arows.append(dict(method=fam, limit=lk, carrier_set=tag, **a))
    pd.DataFrame(arows).round(4).to_csv(os.path.join(tab_dir, "Table16j_carrier_set_ablation.csv"), index=False)
    # which FCS Ts does the (Ts, lam_sw) schedule use? (share of map nodes)
    ts_use = pd.DataFrame(lut_rows)
    ts_use = ts_use[ts_use.method == "FCS-MPCC (Ts, lam_sw)"].groupby(["limit", "Ts_us"]).size().rename("nodes").reset_index()
    ts_use.to_csv(os.path.join(tab_dir, "Table16k_FCS_Ts_usage.csv"), index=False)

    # ------------------------------------------------------------------ 4. held-out closed-loop validation of the LUTs
    print("[6c] held-out validation of the schedules (off-grid points, fresh seeds)")
    ho_raw = vsf.build_maps(cfg, HO_RPM, HO_T, [100, 101, 102], record=record, tag="S10_vsf_heldout")
    ho = vsf.average_maps(ho_raw)
    ho["P_loss"] = vsf.ploss(ho)
    hrows = []
    for lk, (ldesc, lfn) in LIMITS.items():
        for meth, fams in METHODS_ITEMS:
            sel = vsf.select(dfm, fams, lfn)
            lut = vsf.make_lut(sel, rg, tg, ids)
            pl, pr, fe, ol, orr, ofe = [], [], [], [], [], []
            for (rpm, tq), gp in ho.groupby(["rpm", "torque"], sort=False):
                fam, lab = vsf.lookup(lut, ids, rg, tg, rpm, tq)
                x = gp[(gp.family == fam) & (gp.label == lab)].iloc[0]
                lim = lfn(rpm, tq)
                pl.append(x.P_loss)
                pr.append(x.T_rms)
                fe.append(bool(x.T_rms <= lim and not x.track_fail))
                so = vsf.select(gp, fams, lfn)
                ol.append(so.P_loss.iloc[0])
                orr.append(so.T_rms.iloc[0])
                ofe.append(bool(so.feasible.iloc[0]))
            hrows.append(dict(limit=lk, method=BASE_NAME[meth], points=len(pl), LUT_loss_W=np.mean(pl),
                              oracle_loss_W=np.mean(ol), LUT_regret_pct=100 * (np.mean(pl) / np.mean(ol) - 1),
                              LUT_ripple_Nm=np.mean(pr), oracle_ripple_Nm=np.mean(orr),
                              LUT_feasible_frac=np.mean(fe), oracle_feasible_frac=np.mean(ofe)))
    pd.DataFrame(hrows).round(4).to_csv(os.path.join(tab_dir, "Table16g_heldout_validation.csv"), index=False)
    ho_raw.to_csv(os.path.join(tab_dir, "vsf_heldout_all_seeds.csv"), index=False)

    # ------------------------------------------------------------------ 5. dynamic cost of the lower carriers (iq step)
    print("[6c] iq-step response vs carrier (PI bandwidth = min(800 Hz, fc/12))")
    srows = []
    sc_cache = {}
    for fam, cid in vsf.PWM_FAMILIES:
        for tus, fc in zip(vsf.CARRIER_TS_US, vsf.CARRIERS_HZ):
            for sd in seeds:
                Ts = tus * 1e-6
                sc = S.step_current(cfg, Ts)
                r = runner.run(cid, cfg, sc, seed=sd, Ts=Ts, ctrl_overrides={"nsub": int(round(Ts / vsf.H_BASE))})
                if record:
                    record(r, "S10_vsf_step")
                ts, os_ = step_response(r, sc["t_event"])
                srows.append(dict(family=fam, carrier_kHz=fc / 1e3, seed=sd, t_settle_ms=ts, overshoot_pct=os_,
                                  bandwidth_Hz=r["cp"][2] / cfg["motor"]["Ld"] / (2 * np.pi), ok=r["ok"]))
    ts_ = pd.DataFrame(srows)
    ts_.groupby(["family", "carrier_kHz"], sort=False).mean(numeric_only=True).drop(columns=["seed", "ok"]).round(3).to_csv(
        os.path.join(tab_dir, "Table16i_iq_step_vs_carrier.csv"))

    # ------------------------------------------------------------------ 6. figures
    P.pareto_figure(tfix, tsw, t16[(t16.loss_case == "nominal") & (t16.kind == K_IDEAL) & (t16.limit.isin(["abs3", "abs1", "rel5"]))], fig_dir, "Fig9_Pareto_loss_vs_ripple_WLTC")
    P.schedule_maps_figure(pd.DataFrame(lut_rows), rg, tg, fig_dir, "Fig10_VSF_carrier_LUT")
    n_all = len(raw) + len(ho_raw) + len(ts_)
    print(f"[6c] done in {time.time() - t0:.0f} s ({n_all} runs, {n_failed_maps + int((~ho_raw.ok).sum()) + int((~ts_.ok).sum())} non-finite)")
    return dict(n_runs=n_all, n_nonfinite=n_failed_maps + int((~ho_raw.ok).sum()) + int((~ts_.ok).sum()))


METHODS_ITEMS = list(vsf.METHODS.items())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    cfg = params.load_config()
    RES = os.path.join(params.ROOT, "results")
    tab, fig = os.path.join(RES, "tables"), os.path.join(RES, "figures")
    os.makedirs(tab, exist_ok=True)
    os.makedirs(fig, exist_ok=True)
    run(cfg, [200.0, 500.0, 1000.0, 2000.0, 3000.0, 4500.0, 6000.0, 7500.0, 9500.0],
        [5.0, 15.0, 30.0, 60.0, 100.0, 150.0, 200.0, 300.0, 430.0], [0, 1] if a.quick else [0, 1, 2], tab, fig,
        quick=a.quick)
