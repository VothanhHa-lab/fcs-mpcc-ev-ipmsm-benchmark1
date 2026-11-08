"""Stator iron loss with PWM harmonics, computed from the simulated waveforms.

Model (harmonic-aware Bertotti / Steinmetz-Bertotti, per unit mass, applied to stator tooth and yoke)

    p = k_h f B^alpha  +  k_c (f B)^2  +  k_e (f B)^1.5          (sinusoidal flux, alpha = 2 here)

generalised to non-sinusoidal flux as
    hysteresis, fundamental : k_h f_e B1^2
    hysteresis, harmonics   : k_h  sum_k |f_k| B_k^2               (per-harmonic Steinmetz; Parseval form, alpha = 2)
    eddy (classical)        : k_c/(2 pi^2) < (dB/dt)^2 >           (time domain, exact from dpsi/dt = v - Rs i)
    excess                  : k_e/C_e      < |dB/dt|^1.5 >         (time domain, C_e = (2 pi)^1.5 * <|cos|^1.5> = 8.763)
with the lamination skin-effect screening factor F(d/delta) applied to the PWM harmonic part of the eddy and excess terms
(see `screen_factor`).

Flux density from the simulated flux linkage (see README and configs/default.json "iron"):
    psi_d = Ld i_d + psi_f ,  psi_q = Lq i_q   (TRUE plant currents and plant inductances)
    stator-frame flux-linkage derivative  dpsi/dt = v - Rs i   (applied switched voltage incl. dead time, evaluated at
                                                                every 0.25 us sub-step inside core.simulate; |.| is frame invariant)
    local B at angular position phi of the stator = K * (psi . u_phi), K_tooth, K_yoke from geometry;
    spatial average over phi:  <(dB/dt)^2>_phi = K^2 |dpsi/dt|^2 / 2 ,   <|dB/dt|^1.5>_phi = K^1.5 c15 |dpsi/dt|^1.5 ,
    c15 = <|cos|^1.5> = 0.5564.  For a pure rotating flux of amplitude B1 these reduce exactly to the Bertotti formulas.

What is modelled : stator tooth + yoke iron loss (hysteresis, classical eddy, excess), fundamental and PWM harmonics.
What is NOT      : rotor core and magnet eddy losses, stray/slot-leakage-flux losses, saturation of the flux-to-B map
                   (B is linear in psi, so B is over-estimated near peak load), rotational-field hysteresis excess,
                   temperature dependence, FEA field distribution (tooth tips, slot harmonics).
All ripple flux linkage (psi = L i, which includes leakage) is assumed to pass through the stator iron: an UPPER BOUND on the
harmonic loss; the skin-effect factor only partly compensates (see README).
"""
from dataclasses import dataclass, field, replace
import numpy as np
from scipy.special import gamma

from . import core

C15 = float(gamma(1.25) / (np.sqrt(np.pi) * gamma(1.75)))     # <|cos x|^1.5> over a period = 0.5564
CE = (2 * np.pi) ** 1.5 * C15                                   # Bertotti excess constant, 8.7633


@dataclass
class IronParams:
    # material (per kg)
    k_h: float
    k_c: float
    k_e: float
    alpha: float
    build_factor: float
    thickness: float
    sigma: float
    mu_r_eff: float
    # geometry-derived
    K_t: float          # T per Wb  (peak tooth B / peak stator flux linkage)
    K_y: float          # T per Wb
    m_t: float          # kg tooth iron
    m_y: float          # kg yoke iron
    # scenario switches (sensitivity)
    harm_mult: float = 1.0      # multiplies the whole harmonic (PWM) part
    screening: bool = True      # lamination skin-effect screening of harmonic eddy / excess
    resolved_only: bool = False # count only the part of the PWM spectrum resolved by the 400 kHz log (lower bound where under-resolved)
    info: dict = field(default_factory=dict)

    @staticmethod
    def from_cfg(cfg):
        ic = cfg["iron"]
        g, w, m = ic["geometry"], ic["winding"], ic["material"]
        assert abs(m["alpha"] - 2.0) < 1e-12, "harmonic hysteresis sum implemented for alpha = 2 only"
        Ns = w["coils_per_phase"] * w["turns_per_coil"] / w["parallel_paths"]       # series turns per phase
        Nkw = Ns * w["k_w1"]
        p = cfg["motor"]["p"]
        Q = g["Q"]
        Lk = g["L_stack"] * g["k_stack"]
        hs = 0.5 * (g["D_out"] - g["D_bore"]) - g["yoke_height"]                     # slot depth
        # peak pole flux per Wb of fundamental flux linkage amplitude: Phi_pole = psi/(Ns kw1)
        # peak tooth flux = (pi/2) Phi_pole * tau_slot/tau_pole = (pi/2) Phi_pole * 2p/Q  (sinusoidal airgap B)
        K_t = (0.5 * np.pi * 2 * p / Q) / (Nkw * g["tooth_width"] * Lk)
        K_y = 1.0 / (Nkw * 2.0 * g["yoke_height"] * Lk)                               # B_y = Phi_pole / (2 h_y L)
        V_t = Q * g["tooth_width"] * hs * Lk
        r_o = 0.5 * g["D_out"]
        V_y = np.pi * (r_o ** 2 - (r_o - g["yoke_height"]) ** 2) * Lk
        psi_f = cfg["motor"]["psi_f"]
        info = dict(N_series_turns_per_phase=Ns, N_kw1=Nkw, slot_depth_mm=1e3 * hs,
                    Phi_pole_noload_mWb=1e3 * psi_f / Nkw,
                    B_gap1_noload_T=psi_f / Nkw / ((2 / np.pi) * (np.pi * 0.5 * (g["D_bore"] + 0.1499) / (2 * p)) * g["L_stack"]),
                    B_tooth_noload_T=K_t * psi_f, B_yoke_noload_T=K_y * psi_f,
                    K_tooth_T_per_Wb=K_t, K_yoke_T_per_Wb=K_y,
                    m_tooth_kg=g["rho_steel"] * V_t, m_yoke_kg=g["rho_steel"] * V_y)
        return IronParams(k_h=m["k_h"], k_c=m["k_c"], k_e=m["k_e"], alpha=m["alpha"], build_factor=m["build_factor"],
                          thickness=m["thickness"], sigma=m["sigma"], mu_r_eff=m["mu_r_eff"],
                          K_t=K_t, K_y=K_y, m_t=g["rho_steel"] * V_t, m_y=g["rho_steel"] * V_y, info=info)

    def scaled(self, k_h=1.0, k_c=1.0, k_e=1.0, build=1.0, harm=1.0, screening=None, resolved_only=None):
        return replace(self, k_h=self.k_h * k_h, k_c=self.k_c * k_c, k_e=self.k_e * k_e,
                       build_factor=self.build_factor * build, harm_mult=self.harm_mult * harm,
                       screening=self.screening if screening is None else screening,
                       resolved_only=self.resolved_only if resolved_only is None else resolved_only)


def screen_factor(f, ip):
    """Lamination eddy-current screening (skin effect), ratio of loss to the classical (uniform-flux) value:
    F = (3/xi) (sinh xi - sin xi)/(cosh xi - cos xi),  xi = d/delta,  delta = 1/sqrt(pi f mu0 mu_r sigma)."""
    f = np.maximum(np.asarray(f, float), 1e-3)
    delta = 1.0 / np.sqrt(np.pi * f * 4e-7 * np.pi * ip.mu_r_eff * ip.sigma)
    xi = np.minimum(ip.thickness / delta, 30.0)
    x = np.where(xi < 1e-3, 1e-3, xi)
    F = 3.0 / x * (np.sinh(x) - np.sin(x)) / (np.cosh(x) - np.cos(x))
    F = np.where(xi < 1e-3, 1.0, F)
    return np.where(ip.thickness / delta > 30.0, 3.0 / (ip.thickness / delta), F)


def waveform_stats(res, t0=0.01, t1=None, ip=None, cfg=None):
    """Coefficient-independent statistics of the simulated flux-linkage waveform in [t0, t1) (steady operation).
    Returns a dict of scalars (all SI) that `combine` turns into losses for any set of coefficients."""
    fine = res["fine"]
    t = fine[:, core.FT]
    t1 = t[-1] + 1.0 if t1 is None else t1
    m = (t >= t0) & (t < t1)
    pp = res["pp"]
    Ld, Lq, psi_f, p = pp[1], pp[2], pp[3], pp[4]
    dt = float(t[1] - t[0])
    psid = Ld * fine[m, core.FID] + psi_f
    psiq = Lq * fine[m, core.FIQ]
    w_m = float(np.mean(fine[m, core.FW]))
    we = p * w_m
    f_e = abs(we) / (2 * np.pi)
    z = psid + 1j * psiq                       # rotating-frame flux-linkage vector
    z1 = z.mean()
    psi1 = float(abs(z1))
    r = z - z1                                 # ripple (rotating frame)
    N = r.size
    win = np.hanning(N)
    R = np.fft.fft(r * win) / N
    P = np.abs(R) ** 2 / np.mean(win ** 2)     # power per bin, sum ~ <|r|^2>
    nu = np.fft.fftfreq(N, dt)                 # rotating-frame frequency
    fst = np.abs(nu + np.sign(we) * f_e)       # stator-frame frequency of that component
    D2 = float(np.mean(fine[m, core.FDP2]))
    D15 = float(np.mean(fine[m, core.FDP15]))
    D2_1 = (we * psi1) ** 2
    D15_1 = (abs(we) * psi1) ** 1.5
    D2h = max(D2 - D2_1, 0.0)
    D15h = max(D15 - D15_1, 0.0)
    D2_fft = (2 * np.pi) ** 2 * float(np.sum(fst ** 2 * P))
    st = dict(f_e=f_e, psi1=psi1, ripple_rms=float(np.sqrt(np.sum(P))),
              Hh=float(np.sum(fst * P)),                      # sum_k |f_k| |R_k|^2     [Hz Wb^2]
              D2=D2, D15=D15, D2_1=D2_1, D15_1=D15_1, D2h=D2h, D15h=D15h,
              fft_resolved=D2_fft / D2h if D2h > 0 else np.nan)
    if ip is not None:
        F = screen_factor(fst, ip)
        st["rho"] = float(np.sum(fst ** 2 * P * F) / np.sum(fst ** 2 * P)) if D2_fft > 0 else 1.0
        st["f_ripple_c"] = float(np.sum(fst ** 3 * P) / np.sum(fst ** 2 * P)) if D2_fft > 0 else 0.0   # eddy-weighted mean freq
    return st


def combine(st, ip):
    """Iron loss [W] from waveform statistics `st` (see waveform_stats) and coefficients `ip`.
    Fundamental part = loss of the pure rotating fundamental flux; harmonic part = increment due to PWM ripple."""
    rho = st.get("rho", 1.0) if ip.screening else 1.0
    if ip.resolved_only:
        # eddy/excess harmonic content above the Nyquist frequency (200 kHz) of the log is dropped (it is heavily screened
        # in reality); a LOWER bound where fft_resolved < 1 (narrow PWM pulses at low modulation index)
        rho = rho * min(st.get("fft_resolved", 1.0), 1.0)
    out = dict(P_fe=0.0, P_h=0.0, P_c=0.0, P_e=0.0, P_fund=0.0, P_harm=0.0, P_tooth=0.0, P_yoke=0.0,
               P_h_harm=0.0, P_c_harm=0.0, P_e_harm=0.0)
    for K, mass, tag in ((ip.K_t, ip.m_t, "tooth"), (ip.K_y, ip.m_y, "yoke")):
        fe = st["f_e"]
        h1 = ip.k_h * fe * (K * st["psi1"]) ** 2
        c1 = ip.k_c * K ** 2 * st["D2_1"] / (4 * np.pi ** 2)
        e1 = ip.k_e / CE * C15 * K ** 1.5 * st["D15_1"]
        hh = ip.k_h * K ** 2 * st["Hh"] * ip.harm_mult
        ch = ip.k_c * K ** 2 * st["D2h"] * rho / (4 * np.pi ** 2) * ip.harm_mult
        eh = ip.k_e / CE * C15 * K ** 1.5 * st["D15h"] * rho ** 0.75 * ip.harm_mult
        s = ip.build_factor * mass
        out["P_h"] += s * (h1 + hh)
        out["P_c"] += s * (c1 + ch)
        out["P_e"] += s * (e1 + eh)
        out["P_fund"] += s * (h1 + c1 + e1)
        out["P_harm"] += s * (hh + ch + eh)
        out["P_h_harm"] += s * hh
        out["P_c_harm"] += s * ch
        out["P_e_harm"] += s * eh
        out["P_" + tag] = s * (h1 + c1 + e1 + hh + ch + eh)
    out["P_fe"] = out["P_h"] + out["P_c"] + out["P_e"]
    out["harm_share"] = out["P_harm"] / out["P_fe"] if out["P_fe"] > 0 else 0.0
    return out


def iron_loss(res, cfg, t0=0.01, t1=None, ip=None):
    """P_fe [W] and breakdown for a run result: dict with P_fe, P_h, P_c, P_e (hysteresis/eddy/excess), P_fund, P_harm,
    P_h_harm, P_c_harm, P_e_harm, P_tooth, P_yoke, harm_share, plus the waveform statistics under key 'stats'."""
    ip = ip or IronParams.from_cfg(cfg)
    st = waveform_stats(res, t0, t1, ip)
    out = combine(st, ip)
    out["stats"] = st
    return out


STAT_COLS = ["f_e", "psi1", "ripple_rms", "Hh", "D2", "D15", "D2_1", "D15_1", "D2h", "D15h", "fft_resolved", "rho", "f_ripple_c"]


# ----------------------------------------------------------------------------- table helpers
def run_columns(res, cfg, ip=None, t0=0.01, t1=None):
    """Flat dict for CSV rows: nominal iron-loss breakdown [W] + the waveform statistics (prefix ws_) from which any
    coefficient sensitivity can be recomputed without re-simulating (see `combine_row`)."""
    ip = ip or IronParams.from_cfg(cfg)
    st = waveform_stats(res, t0, t1, ip)
    o = combine(st, ip)
    row = {"P_fe": o["P_fe"], "P_fe_fund": o["P_fund"], "P_fe_harm": o["P_harm"], "P_fe_h": o["P_h"], "P_fe_c": o["P_c"],
           "P_fe_e": o["P_e"], "P_fe_h_harm": o["P_h_harm"], "P_fe_c_harm": o["P_c_harm"], "P_fe_e_harm": o["P_e_harm"],
           "P_fe_tooth": o["P_tooth"], "P_fe_yoke": o["P_yoke"],
           "B_tooth_pk_T": ip.K_t * st["psi1"], "B_yoke_pk_T": ip.K_y * st["psi1"]}
    row.update({"ws_" + k: st[k] for k in STAT_COLS})
    return row


def combine_row(row, ip):
    """Recompute the iron-loss breakdown for other coefficients from a CSV row written by `run_columns`."""
    st = {k: float(row["ws_" + k]) for k in STAT_COLS}
    return combine(st, ip)
