"""Figure helpers (IEEE single-column style). Figures are drawn only from run results."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import core, metrics as M  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "lines.linewidth": 0.8, "axes.grid": True, "grid.alpha": 0.3,
                     "savefig.dpi": 300, "pdf.fonttype": 42})
COL = {"PI-SVPWM": "#1f77b4", "Deadbeat-SVPWM": "#2ca02c", "FCS-MPCC (1,1)": "#ff7f0e",
       "FCS-MPCC best fixed": "#8c564b", "AI-FCS-MPCC (paper)": "#d62728",
       "FCS-MPCC lam_sw=0": "#ff7f0e", "FCS-MPCC best fixed lam_sw": "#8c564b",
       "FCS-MPCC + LUT lam_sw": "#9467bd", "FCS-MPCC + NN lam_sw": "#d62728",
       "PI-SVPWM 10 kHz": "#1f77b4"}


def panel_label(ax, k, x=-0.13, y=1.02):
    """Write (a), (b), ... at the upper left of an axis."""
    ax.text(x, y, "(%s)" % "abcdefghij"[k], transform=ax.transAxes, fontsize=8, fontweight="bold",
            va="bottom", ha="left")


def save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, name + ".png"), bbox_inches="tight")
    plt.close(fig)


def step_panels(runs, labels, outdir, name, t_zoom=(0.020, 0.045)):
    n = len(runs)
    fig, axs = plt.subplots(n, 1, figsize=(3.5, 1.15 * n + 0.3), sharex=True)
    for ax, r, lab in zip(axs, runs, labels):
        f = r["fine"]
        t = f[:, core.FT] * 1e3
        m = (f[:, core.FT] >= t_zoom[0]) & (f[:, core.FT] <= t_zoom[1])
        ax.plot(t[m], f[m, core.FIQ], color=COL.get(lab, "k"), lw=0.5, label=r"$i_q$ (true)")
        ia = M.iq_avg(r)
        tk = np.arange(ia.size) * r["Ts"] * 1e3
        mk = (tk >= t_zoom[0] * 1e3) & (tk <= t_zoom[1] * 1e3)
        ax.plot(tk[mk], ia[mk], color="k", lw=0.7, label=r"$\bar i_q$ (per $T_s$)")
        lg = r["log"]
        ax.plot(lg[:, core.LT] * 1e3, lg[:, core.LIQREF], "k--", lw=0.6, label=r"$i_q^*$")
        ax.set_ylabel(r"$i_q$ (A)")
        ax.set_title(lab, fontsize=7, pad=2)
        ax.set_xlim(t_zoom[0] * 1e3, t_zoom[1] * 1e3)
        ax.legend(loc="lower right", ncol=3, fontsize=5.5, framealpha=0.85)
    for k, ax in enumerate(axs):
        panel_label(ax, k, x=-0.16)
    axs[-1].set_xlabel("Time (ms)")
    fig.tight_layout(h_pad=0.3)
    save(fig, outdir, name)


def vehicle_panels(runs, labels, outdir, name, t_lim=None):
    fig, axs = plt.subplots(3, 1, figsize=(3.5, 4.0), sharex=True)
    for r, lab in zip(runs, labels):
        f = r["fine"]
        t = f[:, core.FT]
        axs[0].plot(t, f[:, core.FW] * 60 / (2 * np.pi), color=COL.get(lab, "k"), label=lab)
        n = max(1, int(round(1e-3 / (t[1] - t[0]))))   # 1-ms moving average
        ker = np.ones(n) / n
        axs[1].plot(t[n - 1:], np.convolve(f[:, core.FIQ], ker, "valid"), color=COL.get(lab, "k"), lw=0.7)
        axs[2].plot(t[n - 1:], np.convolve(f[:, core.FTE], ker, "valid"), color=COL.get(lab, "k"), lw=0.7)
    lg = runs[0]["log"]
    axs[0].plot(lg[:, core.LT], lg[:, core.LWREF] * 60 / (2 * np.pi), "k--", lw=0.6, label="reference")
    axs[0].set_ylabel("Speed (rpm)")
    axs[1].set_ylabel(r"$i_q$, 1-ms avg (A)")
    axs[2].set_ylabel(r"$T_e$, 1-ms avg (N m)")
    axs[2].set_xlabel("Time (s)")
    axs[0].legend(fontsize=6, loc="best")
    from matplotlib.lines import Line2D
    hh = [Line2D([0], [0], color=COL.get(l, "k"), lw=0.9, label=l) for l in labels]
    axs[1].legend(handles=hh, fontsize=5, loc="best", ncol=2, title=r"$i_q$ (1-ms average)", title_fontsize=5.5)
    axs[2].legend(handles=hh, fontsize=5, loc="best", ncol=2, title=r"$T_e$ (1-ms average)", title_fontsize=5.5)
    for k, ax in enumerate(axs):
        panel_label(ax, k, x=-0.17)
    if t_lim:
        axs[2].set_xlim(*t_lim)
    fig.tight_layout(h_pad=0.3)
    save(fig, outdir, name)


def torque_zoom(runs, labels, outdir, name, t0, dur=0.002):
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    for r, lab in zip(runs, labels):
        f = r["fine"]
        m = (f[:, core.FT] >= t0) & (f[:, core.FT] < t0 + dur)
        ax.plot((f[m, core.FT] - t0) * 1e3, f[m, core.FTE], color=COL.get(lab, "k"), label=lab, lw=0.6)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(r"$T_e$ (N m)")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    save(fig, outdir, name)


# ---------------------------------------------------------------------------- M1: ripple-constrained comparison
FAM_COL = {"PI-SVPWM": "#1f77b4", "PI-DPWM1": "#2ca02c", "PI-DPWM-I": "#17becf",
           "FCS Ts=10us": "#ff7f0e", "FCS Ts=20us": "#bcbd22", "FCS Ts=50us": "#8c564b"}
METH_COL = {"PI-SVPWM": "#1f77b4", "PI-DPWM1": "#2ca02c", "PI-DPWM-I": "#17becf",
            "FCS-MPCC (Ts = 10 us)": "#d62728", "FCS-MPCC (Ts, lam_sw)": "#7f2a2a"}
LIM_MK = {"abs3": ("o", "3 % abs (12.9 N m)"), "abs1": ("s", "1 % abs (4.3 N m)"), "rel5": ("^", "5 % rel (floor 3 N m)")}


def pareto_figure(tfix, tsw, tsched, outdir, name):
    """WLTC drive loss vs WLTC time-mean torque-ripple RMS.
    thin dotted lines + small markers: every fixed setting (carrier or lam_sw); settings that fail torque
    tracking at some map node are omitted.  thick lines: scheduled frontier (limit sweep).  big markers: the
    three ripple limits of Table 16."""
    fig, ax = plt.subplots(figsize=(3.5, 3.3))
    for fam, g in tfix.groupby("family", sort=False):
        g = g[g.nodes_failing_torque_tracking == 0]
        ax.plot(g.ripple_mean_Nm, g.loss_Wh_per_km, ":", marker=".", ms=3, lw=0.6, color=FAM_COL[fam], alpha=0.8,
                label=None)
    for meth, g in tsw.groupby("method", sort=False):
        g = g.sort_values("limit_Nm")
        ax.plot(g.ripple_mean_Nm, g.loss_Wh_per_km, "-", lw=1.3, color=METH_COL[meth], alpha=0.9)
    for meth, g in tsched.groupby("method", sort=False):
        for r in g.itertuples():
            ax.plot(r.ripple_mean_Nm, r.loss_Wh_per_km, LIM_MK[r.limit][0], ms=5.5, mfc=METH_COL[meth], mec="k", mew=0.6,
                    zorder=5)
    from matplotlib.lines import Line2D
    h1 = [Line2D([0], [0], color=METH_COL[m], lw=1.3, label=m + " (sched.)") for m in METH_COL]
    h2 = [Line2D([0], [0], color="0.4", ls=":", marker=".", ms=3, lw=0.6, label="fixed settings (colour = family)")]
    h3 = [Line2D([0], [0], marker=mk, color="w", mfc="0.7", mec="k", ms=5, label=lab) for mk, lab in LIM_MK.values()]
    ax.legend(handles=h1 + h2 + h3, fontsize=5.2, loc="upper right", ncol=1, framealpha=0.9, handlelength=1.6)
    ax.set_xscale("log")
    ax.set_xlabel("WLTC time-mean torque-ripple RMS (N m)")
    ax.set_ylabel("WLTC drive loss (Wh/km)")
    fig.tight_layout()
    save(fig, outdir, name)


def schedule_maps_figure(lut, rg, tg, outdir, name):
    """Selected carrier (PI-SVPWM VSF) and selected lam_sw (FCS, Ts = 10 us) over the (speed, torque) grid."""
    import matplotlib.colors as mc
    from . import vsf
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.0), sharex=True, sharey=True)
    ri = {v: i for i, v in enumerate(rg)}
    ti = {v: j for j, v in enumerate(tg)}
    titles = {"abs3": "limit 12.9 N m", "abs1": "limit 4.3 N m", "rel5": "limit 5 % of torque (>= 3 N m)"}
    car = list(vsf.CARRIERS_HZ)
    cmap1 = plt.get_cmap("viridis", len(car))
    lams = list(vsf.LAM_GRID)
    cmap2 = plt.get_cmap("magma_r", len(lams))
    for c, lk in enumerate(["abs3", "abs1", "rel5"]):
        for r, (meth, key, vals, cm, lab) in enumerate([
                ("PI-SVPWM", "carrier_kHz", car, cmap1, "carrier (kHz)"),
                ("FCS-MPCC (Ts = 10 us)", "lam_sw", lams, cmap2, r"$\lambda_{sw}$")]):
            g = lut[(lut.limit == lk) & (lut.method == meth)]
            Z = np.full((len(tg), len(rg)), np.nan)
            Fz = np.zeros_like(Z, dtype=bool)
            for x in g.itertuples():
                v = getattr(x, key) * (1e3 if key == "carrier_kHz" else 1.0)
                Z[ti[x.torque], ri[x.rpm]] = int(np.argmin(np.abs(np.array(vals) - v)))
                Fz[ti[x.torque], ri[x.rpm]] = not x.feasible
            ax = axs[r, c]
            im = ax.imshow(np.ma.masked_invalid(Z), origin="lower", cmap=cm, vmin=-0.5, vmax=len(vals) - 0.5,
                           aspect="auto")
            for (jj, ii) in zip(*np.where(Fz)):
                ax.plot(ii, jj, "kx", ms=4)
            ax.set_xticks(range(len(rg)))
            ax.set_xticklabels([f"{v/1e3:g}" for v in rg], fontsize=6)
            ax.set_yticks(range(len(tg)))
            ax.set_yticklabels([f"{v:g}" for v in tg], fontsize=6)
            ax.grid(False)
            if r == 0:
                ax.set_title(titles[lk], fontsize=7)
            if r == 1:
                ax.set_xlabel("Speed (krpm)")
            if c == 0:
                ax.set_ylabel("Torque (N m)")
    for r, (vals, cm, lab, fmt) in enumerate([(car, cmap1, "Carrier (kHz)", lambda v: f"{v/1e3:.3g}"),
                                              (lams, cmap2, r"$\lambda_{sw}$", lambda v: f"{v:g}")]):
        sm = plt.cm.ScalarMappable(cmap=cm, norm=mc.Normalize(-0.5, len(vals) - 0.5))
        cb = fig.colorbar(sm, ax=axs[r, :], ticks=range(len(vals)), pad=0.01, fraction=0.025)
        cb.ax.set_yticklabels([fmt(v) for v in vals], fontsize=6)
        cb.set_label(lab, fontsize=7)
    fig.text(0.01, 0.99, "x = no setting meets the limit (min-ripple fallback)", fontsize=6, va="top")
    save(fig, outdir, name)
