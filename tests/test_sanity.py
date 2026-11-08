"""Physics / numerics sanity checks. Run: python3 tests/test_sanity.py"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from aifcs import core, params, scenarios as S, runner, metrics as M  # noqa: E402
from aifcs import ironloss as IL  # noqa: E402

cfg = params.load_config()
Ts = cfg["control"]["Ts"]
out = []


def check(name, cond, info=""):
    out.append((name, bool(cond), info))
    print(("PASS " if cond else "FAIL ") + name + "  " + info)


# 1) Back-EMF / steady voltage: PI at no load, 1000 rpm, i=0 -> commanded |v| ~ we*psi
sc = S.steady(cfg, Ts, 1000.0, 0.0, T=0.02)
r = runner.run(core.PI, cfg, sc, seed=0, ctrl_overrides={"sigma_i": 0.0, "sigma_w": 0.0})
we = cfg["motor"]["p"] * 1000 * 2 * np.pi / 60
fine = r["fine"]
# reconstruct true vq from dq model at steady state: vq = Rs iq + we(Ld id + psi)
emf = we * cfg["motor"]["psi_f"]
check("back-EMF magnitude (V)", abs(emf - 24.25) < 0.1, f"we*psi={emf:.2f} V (MXP: 42/sqrt3 per krpm = 24.25 V)")

# 2) Energy balance in steady state: P_elec ~ P_cu + P_mech (magnetic energy constant on average)
sc = S.steady(cfg, Ts, 1500.0, 200.0, T=0.04)
r = runner.run(core.PI, cfg, sc, seed=0)
L = M.losses(r)
bal = L["P_elec"] - L["P_cu"] - L["P_mech"]
check("power balance P_elec - P_cu - P_mech", abs(bal) < 0.01 * L["P_elec"],
      f"P_elec={L['P_elec']:.0f} W, residual={bal:.1f} W")

# 3) Torque from currents: mean Te equals requested torque (PI tracks the MTPA currents)
tr = M.torque_ripple(r, 0.02, 0.04)
check("mean torque at 200 N m request", abs(tr["T_mean"] - 200) < 3, f"T_mean={tr['T_mean']:.1f}")

# 4) Time-step convergence (nsub 200 vs 400) on the FCS metrics
sc = S.steady(cfg, Ts, 1500.0, 200.0, T=0.04)
a = runner.run(core.FCS, cfg, sc, seed=3)
b = runner.run(core.FCS, cfg, sc, seed=3, ctrl_overrides={"nsub": 400})
ta, tb = M.torque_ripple(a, 0.02, 0.04), M.torque_ripple(b, 0.02, 0.04)
check("h-convergence FCS T_rms (nsub 200 vs 400)", abs(ta["T_rms"] - tb["T_rms"]) < 0.1 * tb["T_rms"],
      f"{ta['T_rms']:.2f} vs {tb['T_rms']:.2f} N m")
pa, pb = runner.run(core.PI, cfg, sc, seed=3), runner.run(core.PI, cfg, sc, seed=3, ctrl_overrides={"nsub": 400})
ra, rb = M.torque_ripple(pa, 0.02, 0.04), M.torque_ripple(pb, 0.02, 0.04)
check("h-convergence PI T_rms (nsub 200 vs 400)", abs(ra["T_rms"] - rb["T_rms"]) < 0.15 * rb["T_rms"],
      f"{ra['T_rms']:.3f} vs {rb['T_rms']:.3f} N m")

# 5) Independent runs: same seed -> identical; different seeds -> different noise
c1 = runner.run(core.FCS, cfg, sc, seed=5)
c2 = runner.run(core.FCS, cfg, sc, seed=5)
c3 = runner.run(core.FCS, cfg, sc, seed=6)
check("determinism (same seed)", np.array_equal(c1["fine"], c2["fine"]))
check("seed changes noise", not np.array_equal(c1["fine"], c3["fine"]))

# 6) Scaling invariance of the 2-term cost: (lam_d, lam_q)=(2,2) must equal (1,1)
d1 = runner.run(core.FCS, cfg, sc, seed=7, ctrl_overrides={"lam_d": 1.0, "lam_q": 1.0})
d2 = runner.run(core.FCS, cfg, sc, seed=7, ctrl_overrides={"lam_d": 2.0, "lam_q": 2.0})
check("argmin invariant to common scaling of (lam_d, lam_q)",
      np.array_equal(d1["log"][:, core.LSTATE], d2["log"][:, core.LSTATE]))

# 7) Metric hand checks
fake = dict(fine=np.zeros((4, core.NFINE)), log=np.zeros((2, core.NLOG)), Ts=1.0, cp=np.array([0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
fake["fine"][:, core.FT] = [0.0, 0.5, 1.0, 1.5]
fake["fine"][:, core.FIQ] = [1.0, 3.0, 2.0, 2.0]
fake["log"][:, core.LIQREF] = [2.0, 2.0]
tk = M.tracking(fake, 0.0, 2.0)
check("RMSE hand check", abs(tk["RMSE_iq"] - np.sqrt(0.5)) < 1e-12, f"{tk['RMSE_iq']:.4f} vs 0.7071")
check("iq_avg hand check", np.allclose(M.iq_avg(fake), [2.0, 2.0]))

# 8) Vehicle speed scenario reaches the reference; acceleration bounded by Te_max / J
sc = S.speed_step(cfg, Ts, T=1.0)
r = runner.run(core.PI, cfg, sc, seed=0)
w_end = r["fine"][-1, core.FW] * 60 / (2 * np.pi)
J = sc["J"]
tmin = J * (700 * 2 * np.pi / 60) / 430.0
sm = M.speed_metrics(r, sc["t_event"])
check("speed step reaches 1200 rpm", abs(w_end - 1200) < 5, f"end={w_end:.1f} rpm")
check("settling not faster than physics", sm["t_recover_ms"] / 1e3 > tmin,
      f"t_rec={sm['t_recover_ms']:.0f} ms > t_min={1e3*tmin:.0f} ms")

# 9) Field weakening: torque is tracked deep in the constant-power region (10 000 rpm, 150 N m)
op = params.op_point(cfg, 10000.0, 150.0)
sc = S.steady(cfg, Ts, 10000.0, 150.0, T=0.02)
r = runner.run(core.PI, cfg, sc, seed=0)
tr = M.torque_ripple(r, 0.01, 0.02)
m_ = cfg["motor"]
we = m_["p"] * 10000 * 2 * np.pi / 60
vmag = we * np.hypot(m_["Ld"] * op[0] + m_["psi_f"], m_["Lq"] * op[1])
check("FW: |v| needed <= Vdc/sqrt3", vmag <= cfg["inverter"]["Vdc"] / np.sqrt(3), f"{vmag:.0f} V, id*={op[0]:.0f} A")
check("FW: 150 N m tracked at 10 000 rpm (PI)", abs(tr["T_mean"] - 150) < 5, f"T_mean={tr['T_mean']:.1f}")
r = runner.run(core.FCS, cfg, sc, seed=0, Ts=10e-6, ctrl_overrides={"nsub": 40})
tr = M.torque_ripple(r, 0.01, 0.02)
check("FW: 150 N m tracked at 10 000 rpm (FCS, Ts=10us)", abs(tr["T_mean"] - 150) < 5, f"T_mean={tr['T_mean']:.1f}")

# 10) DPWM (M1 baselines) ------------------------------------------------------------------
from aifcs import vsf  # noqa: E402
Ts_d = 100e-6
sc = S.steady(cfg, Ts_d, 1500.0, 200.0, T=0.05)
ovd = {"nsub": int(round(Ts_d / 0.25e-6))}
rp = runner.run(core.PI, cfg, sc, seed=1, Ts=Ts_d, ctrl_overrides=ovd)
rd1 = runner.run(core.DPWM1, cfg, sc, seed=1, Ts=Ts_d, ctrl_overrides=ovd)
rdi = runner.run(core.DPWMI, cfg, sc, seed=1, Ts=Ts_d, ctrl_overrides=ovd)
for nm, rr in [("DPWM1", rd1), ("DPWM-I", rdi)]:
    ratio = rr["ncomm"].sum() / rp["ncomm"].sum()
    check(f"{nm}: commutations ~ 2/3 of SVPWM at the same carrier", 0.64 < ratio < 0.72, f"ratio={ratio:.3f}")
    tm = M.torque_ripple(rr, 0.01, 0.05)["T_mean"]
    check(f"{nm}: mean torque tracks the 200 N m request", abs(tm - 200) < 3, f"T_mean={tm:.1f}")
check("DPWM-I (current-based clamp) switching loss <= DPWM1 (voltage-based)",
      M.losses(rdi)["P_sw"] <= M.losses(rd1)["P_sw"], f"{M.losses(rdi)['P_sw']:.0f} vs {M.losses(rd1)['P_sw']:.0f} W")
sc_fw = S.steady(cfg, 50e-6, 10000.0, 150.0, T=0.02)
r = runner.run(core.DPWM1, cfg, sc_fw, seed=0)
tm = M.torque_ripple(r, 0.01, 0.02)["T_mean"]
check("DPWM1: 150 N m tracked at 10 000 rpm (field weakening)", abs(tm - 150) < 5, f"T_mean={tm:.1f}")


def _ideal_ripple(mode, Vamp, Vdc=360.0, n_ang=60, nf=800):
    """Ideal volt-second ripple (no delay, no dead time, no noise) of centre-aligned SVPWM / DPWM1, RMS over a cycle."""
    out = []
    for th in np.linspace(0, 2 * np.pi, n_ang, endpoint=False):
        v = np.array([Vamp * np.cos(th), Vamp * np.cos(th - 2 * np.pi / 3), Vamp * np.cos(th + 2 * np.pi / 3)])
        if mode == "svpwm":
            vo = -0.5 * (v.max() + v.min())
        else:
            vo = (0.5 * Vdc - v.max()) if v.max() >= -v.min() else (-0.5 * Vdc - v.min())
        d = 0.5 + (v + vo) / Vdc
        t = (np.arange(nf) + 0.5) / nf
        g = (np.abs(2 * t - 1)[None, :] < d[:, None]).astype(float)
        vp = Vdc * (g - 0.5)
        ea = (2 * vp[0] - vp[1] - vp[2]) / 3 - (2 * v[0] - v[1] - v[2]) / 3
        eb = (vp[1] - vp[2]) / np.sqrt(3) - (v[1] - v[2]) / np.sqrt(3)
        ia, ib = np.cumsum(ea) / nf, np.cumsum(eb) / nf
        ia -= ia.mean()
        ib -= ib.mean()
        out.append(np.mean(ia ** 2 + ib ** 2))
    return np.sqrt(np.mean(out))


op = params.op_point(cfg, 1500.0, 200.0)
m_ = cfg["motor"]
vm = 2 * np.pi * 1500 / 60 * m_["p"] * np.hypot(m_["Ld"] * op[0] + m_["psi_f"], m_["Lq"] * op[1])
th_ratio = _ideal_ripple("dpwm", vm) / _ideal_ripple("svpwm", vm)
nz = {"nsub": 400, "sigma_i": 0.0, "sigma_w": 0.0}
a_ = M.torque_ripple(runner.run(core.PI, cfg, sc, seed=1, Ts=Ts_d, ctrl_overrides=nz), 0.01, 0.05)["T_rms"]
b_ = M.torque_ripple(runner.run(core.DPWM1, cfg, sc, seed=1, Ts=Ts_d, ctrl_overrides=nz), 0.01, 0.05)["T_rms"]
check("DPWM1/SVPWM torque-ripple ratio matches the ideal volt-second calculation (15 %)",
      abs(b_ / a_ / th_ratio - 1) < 0.15, f"simulated {b_ / a_:.2f} vs ideal {th_ratio:.2f}")

# 11) PI bandwidth rule at every carrier of the VSF set: wc = 2 pi min(800 Hz, fc / 12)
okbw = True
for tus, fc in zip(vsf.CARRIER_TS_US, vsf.CARRIERS_HZ):
    pp_, mp_, cp_ = params.build(cfg, cfg["motor"]["J_rotor"], Ts=tus * 1e-6)
    bw = cp_[2] / cfg["motor"]["Ld"] / (2 * np.pi)
    okbw &= abs(bw - min(cfg["control"]["pi_bandwidth_hz"], fc / 12)) < 1e-6 * bw
check("PI bandwidth = min(800 Hz, fc/12) for all VSF carriers", okbw)

# 12) VSF LUT returns a carrier from the allowed set (synthetic LUT + the real schedule if it exists)
rg_, tg_ = [200.0, 1000.0, 5000.0], [10.0, 100.0, 400.0]
ids_ = [("PI-SVPWM", vsf.carrier_label(f)) for f in vsf.CARRIERS_HZ]
rng_ = np.random.default_rng(0)
lut_ = rng_.integers(0, len(ids_), size=(3, 3))
qs = [(rng_.uniform(0, 12000), rng_.uniform(0, 600)) for _ in range(200)]
okset = all(vsf.carrier_lookup(lut_, ids_, rg_, tg_, a, b) in vsf.CARRIERS_HZ for a, b in qs)
okset &= all(vsf.carrier_lookup(lut_, ids_, rg_, tg_, rg_[i], tg_[j]) == vsf.CARRIERS_HZ[lut_[i, j]]
             for i in range(3) for j in range(3))
check("VSF LUT lookup returns a carrier of the allowed set (and the node value at nodes)", okset)
fL = os.path.join(params.ROOT, "results", "tables", "Table16d_schedule_LUTs.csv")
if os.path.exists(fL):
    import pandas as pd  # noqa: E402
    L_ = pd.read_csv(fL)
    L_ = L_[L_.family.str.startswith("PI")]
    allowed = {round(f / 1e3, 3) for f in vsf.CARRIERS_HZ}
    check("scheduled carriers in Table16d are all members of the carrier set",
          all(round(c, 3) in allowed for c in L_.carrier_kHz), f"{len(L_)} LUT entries")


# ------------------------------------------------------------------ stator iron-loss model (ironloss.py)
IP = IL.IronParams.from_cfg(cfg)
inf = IP.info
check("iron: series turns/phase = 18 coils x 2 turns / 3 paths = 12", abs(inf["N_series_turns_per_phase"] - 12.0) < 1e-12)
check("iron: no-load peak tooth B in 1.6-1.7 T, yoke in 1.4-1.5 T",
      1.6 <= inf["B_tooth_noload_T"] <= 1.7 and 1.4 <= inf["B_yoke_noload_T"] <= 1.5,
      f"tooth {inf['B_tooth_noload_T']:.3f} T, yoke {inf['B_yoke_noload_T']:.3f} T, airgap fundamental {inf['B_gap1_noload_T']:.2f} T")
check("iron: stator iron mass plausible (10-25 kg)", 10 < IP.m_t + IP.m_y < 25, f"tooth {IP.m_t:.2f} kg + yoke {IP.m_y:.2f} kg")

# (0) the time-domain constants reproduce Bertotti exactly for a pure rotating flux (no ripple)
fe_, psi1_ = 150.0, 0.0772
st = dict(f_e=fe_, psi1=psi1_, Hh=0.0, D2=(2 * np.pi * fe_ * psi1_) ** 2, D15=(2 * np.pi * fe_ * psi1_) ** 1.5,
          D2_1=(2 * np.pi * fe_ * psi1_) ** 2, D15_1=(2 * np.pi * fe_ * psi1_) ** 1.5, D2h=0.0, D15h=0.0, rho=1.0)
o = IL.combine(st, IP)
ref = 0.0
for K, mass in ((IP.K_t, IP.m_t), (IP.K_y, IP.m_y)):
    B = K * psi1_
    ref += IP.build_factor * mass * (IP.k_h * fe_ * B ** 2 + IP.k_c * (fe_ * B) ** 2 + IP.k_e * (fe_ * B) ** 1.5)
check("iron: time-domain eddy/excess constants equal Bertotti k_c(fB)^2 + k_e(fB)^1.5 for sinusoidal flux",
      abs(o["P_fe"] - ref) < 1e-9 * ref, f"{o['P_fe']:.3f} W vs closed form {ref:.3f} W")
check("iron: harmonic multiplier scales only the harmonic part",
      abs(IL.combine(dict(st, D2h=100.0, D15h=10.0, Hh=1e-6), IP.scaled(harm=3))["P_fe"]
          - (IL.combine(dict(st, D2h=100.0, D15h=10.0, Hh=1e-6), IP.scaled(harm=0))["P_fe"]
             + 3 * IL.combine(dict(st, D2h=100.0, D15h=10.0, Hh=1e-6), IP)["P_harm"])) < 1e-9)

def iron_run(ctrl, Ts, rpm, tq, ov=None, seed=0):
    sc_ = S.steady(cfg, Ts, rpm, tq, T=0.04)
    ov_ = dict(ov or {}, nsub=int(round(Ts / 0.25e-6)))
    r_ = runner.run(ctrl, cfg, sc_, seed=seed, Ts=Ts, ctrl_overrides=ov_)
    return r_, IL.iron_loss(r_, cfg, 0.01)

# (1) resolution of dpsi/dt: in-core sub-step statistic vs finite differences of the 2.5 us log
for ctrl_, Ts_, lab_ in [(core.PI, 100e-6, "PI 10 kHz"), (core.FCS, 10e-6, "FCS Ts=10 us")]:
    r_, o_ = iron_run(ctrl_, Ts_, 3000.0, 30.0)
    f_ = r_["fine"]
    m_ = f_[:, core.FT] >= 0.01
    dt_ = f_[1, core.FT] - f_[0, core.FT]
    p_ = cfg["motor"]
    we_ = p_["p"] * float(f_[m_, core.FW].mean())
    psid_ = p_["Ld"] * f_[:, core.FID] + p_["psi_f"]
    exd = p_["Ld"] * np.diff(f_[:, core.FID]) / dt_ - we_ * p_["Lq"] * f_[:-1, core.FIQ]
    exq = p_["Lq"] * np.diff(f_[:, core.FIQ]) / dt_ + we_ * psid_[:-1]
    d2fd = float(np.mean((exd ** 2 + exq ** 2)[m_[:-1]]))
    d2 = o_["stats"]["D2"]
    check(f"dpsi/dt resolution ({lab_}): finite-difference <|dpsi/dt|^2> within 85-100 % of the exact sub-step value",
          0.85 <= d2fd / d2 <= 1.0, f"ratio {d2fd / d2:.3f} (exact {d2:.0f} V^2, 2.5 us finite difference {d2fd:.0f} V^2)")
    check(f"dpsi/dt resolution ({lab_}): FFT of 400 kHz log resolves 90-110 % of harmonic <|dpsi/dt|^2>",
          0.9 <= o_["stats"]["fft_resolved"] <= 1.1, f"{o_['stats']['fft_resolved']:.3f}")

# (a) no load, PI at high carrier: fundamental part equals closed form; ripple flux and minor-loop hysteresis vanish ~ 1/f_sw
r10, o10 = iron_run(core.PI, 100e-6, 1000.0, 0.0)
r40, o40 = iron_run(core.PI, 25e-6, 1000.0, 0.0)
fe1 = 1000 * cfg["motor"]["p"] / 60.0
ref = 0.0
for K, mass in ((IP.K_t, IP.m_t), (IP.K_y, IP.m_y)):
    B = K * cfg["motor"]["psi_f"]
    ref += IP.build_factor * mass * (IP.k_h * fe1 * B ** 2 + IP.k_c * (fe1 * B) ** 2 + IP.k_e * (fe1 * B) ** 1.5)
check("iron: no-load fundamental part equals Bertotti closed form at B = K psi_f (1000 rpm)",
      abs(o40["P_fund"] - ref) < 0.02 * ref, f"{o40['P_fund']:.1f} W vs {ref:.1f} W")
rr = o40["stats"]["ripple_rms"] / o10["stats"]["ripple_rms"]
check("iron: no load, PI 40 kHz vs 10 kHz: ripple flux linkage falls ~ 1/f_sw (ratio 0.15-0.45)", 0.15 < rr < 0.45, f"ratio {rr:.3f}")
hs = o40["P_h_harm"] / o40["P_h"]
check("iron: no load, PI 40 kHz: minor-loop (harmonic) hysteresis < 2 % of hysteresis loss", hs < 0.02, f"{100 * hs:.2f} %")
print(f"INFO  no load 1000 rpm PI 40 kHz: harmonic share of TOTAL iron loss = {100 * o40['harm_share']:.0f} % "
      f"(eddy+excess of the PWM voltage steps do NOT vanish with carrier: <|dpsi/dt|^2> is set by Vdc, not by f_sw)")

# (b) speed dependence of the fundamental iron loss at constant flux (no load): exponent between 1 (hysteresis) and 2 (eddy)
sp = [1000.0, 2000.0, 4000.0]
Pf = [iron_run(core.PI, 50e-6, w_, 0.0)[1]["P_fund"] for w_ in sp]
sl = np.polyfit(np.log(sp), np.log(Pf), 1)[0]
check("iron: fundamental loss grows as f^n with 1 < n < 2 (no load, 1000-4000 rpm)", 1.0 < sl < 2.0 and Pf[0] < Pf[1] < Pf[2],
      f"n = {sl:.2f}, P = {Pf[0]:.0f}/{Pf[1]:.0f}/{Pf[2]:.0f} W")

# (c) fixed operating point (3000 rpm, 30 N m): harmonic iron loss vs switching frequency
pi_sw = [(400e-6, "2.5 kHz"), (100e-6, "10 kHz"), (25e-6, "40 kHz")]
pr = [iron_run(core.PI, t_, 3000.0, 30.0) for t_, _ in pi_sw]
Ph = [x[1]["P_harm"] for x in pr]
check("iron: PI carrier sweep 2.5 -> 10 -> 40 kHz: harmonic iron loss decreases monotonically",
      Ph[0] > Ph[1] > Ph[2], "P_harm = " + " / ".join(f"{v:.0f}" for v in Ph) + " W")
Hh_ = [x[1]["P_h_harm"] for x in pr]
check("iron: PI carrier sweep: minor-loop hysteresis falls ~ 1/f_sw (ratio 2.5 kHz / 40 kHz > 10)", Hh_[0] / Hh_[2] > 10.0,
      "P_h_harm = " + " / ".join(f"{v:.2f}" for v in Hh_) + " W")
fc = [iron_run(core.FCS, t_, 3000.0, 30.0) for t_ in (50e-6, 25e-6, 12.5e-6)]
fsw_fc = [M.fsw(x[0]) for x in fc]
Hf = [x[1]["P_h_harm"] for x in fc]
check("iron: FCS (lam_sw=0) Ts sweep 50 -> 12.5 us: f_sw rises and minor-loop hysteresis falls",
      fsw_fc[0] < fsw_fc[1] < fsw_fc[2] and Hf[0] > Hf[1] > Hf[2],
      "f_sw = " + " / ".join(f"{v / 1e3:.1f}" for v in fsw_fc) + " kHz, P_h_harm = " + " / ".join(f"{v:.2f}" for v in Hf) + " W")
Pt = [x[1]["P_harm"] for x in fc]
check("iron: FCS (lam_sw=0) Ts sweep 50 -> 12.5 us: total harmonic iron loss does not increase", Pt[0] >= Pt[2],
      "P_harm = " + " / ".join(f"{v:.0f}" for v in Pt) + " W")
fl = [iron_run(core.FCS, 10e-6, 3000.0, 30.0, {"lam_sw": l_}) for l_ in (0.0, 1000.0)]
print(f"INFO  FCS Ts=10 us 3000 rpm 30 N m: lam_sw 0 -> 1000 gives f_sw {M.fsw(fl[0][0]) / 1e3:.1f} -> {M.fsw(fl[1][0]) / 1e3:.1f} kHz, "
      f"harmonic iron {fl[0][1]['P_harm']:.0f} -> {fl[1][1]['P_harm']:.0f} W (minor-loop part {fl[0][1]['P_h_harm']:.1f} -> {fl[1][1]['P_h_harm']:.1f} W)")

ok = all(c for _, c, _ in out)
print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
os.makedirs(os.path.join(params.ROOT, "results"), exist_ok=True)
with open(os.path.join(params.ROOT, "results", "sanity_checks.txt"), "w") as f:
    for n, c, i in out:
        f.write(("PASS " if c else "FAIL ") + n + "  " + i + "\n")
