"""Closed-loop PMSM drive simulator (numba-jitted).

Plant  : dq PMSM, two-level VSI with per-phase dead time, integrated with a fixed
         sub-step h = Ts/nsub (forward Euler; convergence checked in tests).
Control: executed once per Ts (zero-order hold), one-period computation delay
         (command computed at t_k is applied on [t_{k+1}, t_{k+2})), measurement
         noise, shared PI speed loop and load-torque observer.

Controller ids
  0 PI      : dq PI + decoupling + SVPWM (min-max injection), carrier period = Ts
  1 DB      : deadbeat current control + SVPWM
  2 FCS     : FCS-MPCC, fixed weights (lam_d, lam_q, lam_sw)
  3 FCS_NN  : FCS-MPCC, (lam_d, lam_q) = NN(e_d, e_q, w_m, TL_hat)   [manuscript method]
  4 FCS_SWNN: FCS-MPCC, lam_sw = NN(w_m, TL_hat)                     [proposed reformulation]
  5 FCS_SWLUT: FCS-MPCC, lam_sw = bilinear LUT(w_m, TL_hat)          [non-learning scheduler]
  6 DPWM1   : same PI current loop as 0, discontinuous PWM. Voltage-based DPWM1 placement:
              in every control period the phase with the largest |v*| is clamped to the DC rail
              of its own sign (60-degree clamp centred on that phase's reference-voltage peak,
              each phase clamped 2 x 60 of 360 electrical degrees -> 2/3 of the SVPWM commutations)
  7 DPWMI   : same as 6, but the clamped phase is chosen from the PHASE CURRENT: the two feasible
              clamps (max-voltage phase -> +rail, min-voltage phase -> -rail) are compared and the
              one whose phase carries the larger |i| (reference dq current rotated to the PWM angle)
              is used, i.e. the clamp is placed at the phase-current peak (minimum switching energy
              for E_sw ~ |i|). Not available from the voltage alone when the current lags the voltage.
Variable switching frequency (VSF) PWM is NOT a separate controller id: the carrier period equals
the control period Ts, which is fixed within one run; scheduling Ts per operating point is done
outside the jitted core (src/aifcs/vsf.py: carrier LUT -> Ts per run).
"""
import numpy as np
from numba import njit

PI, DB, FCS, FCS_NN, FCS_SWNN, FCS_SWLUT, DPWM1, DPWMI = 0, 1, 2, 3, 4, 5, 6, 7
NAMES = {PI: "PI-SVPWM", DB: "Deadbeat-SVPWM", FCS: "FCS-MPCC",
         FCS_NN: "AI-FCS-MPCC (paper)", FCS_SWNN: "FCS-MPCC + NN lam_sw",
         FCS_SWLUT: "FCS-MPCC + LUT lam_sw", DPWM1: "PI-DPWM1", DPWMI: "PI-DPWM-I"}

SQ3 = np.sqrt(3.0)

# indices of the per-Ts log
LT, LIDREF, LIQREF, LIDM, LIQM, LWM, LWREF, LTLH, LLD, LLQ, LLSW, LSTATE, LTL = range(13)
NLOG = 13
# indices of the fine log
# FDP2 / FDP15: mean over the log interval [t, t + log_div*h) of |dpsi/dt|^2 [V^2] and |dpsi/dt|^1.5 [V^1.5],
# where dpsi/dt = v - Rs i is the stator flux-linkage derivative in the stationary frame, evaluated from the
# applied (switched, dead-time-affected) voltage at EVERY plant sub-step (0.25 us), not only at the log instants.
# |dpsi/dt| is frame-invariant, so it is computed from the dq components of (v_dq - Rs i_dq).
FT, FID, FIQ, FIA, FTE, FW, FDP2, FDP15 = range(8)
NFINE = 8


@njit(cache=True)
def nn_forward(x, W1, b1, W2, b2, out):
    nh = W1.shape[0]
    z = np.empty(nh)
    for i in range(nh):
        s = b1[i]
        for j in range(x.shape[0]):
            s += W1[i, j] * x[j]
        z[i] = np.tanh(s)
    for o in range(W2.shape[0]):
        s = b2[o]
        for i in range(nh):
            s += W2[o, i] * z[i]
        out[o] = 1.0 / (1.0 + np.exp(-s))


@njit(cache=True)
def lut_bilinear(gx, gy, V, x, y):
    if x < gx[0]:
        x = gx[0]
    if x > gx[-1]:
        x = gx[-1]
    if y < gy[0]:
        y = gy[0]
    if y > gy[-1]:
        y = gy[-1]
    i = 0
    while i < gx.shape[0] - 2 and x > gx[i + 1]:
        i += 1
    j = 0
    while j < gy.shape[0] - 2 and y > gy[j + 1]:
        j += 1
    tx = (x - gx[i]) / (gx[i + 1] - gx[i])
    ty = (y - gy[j]) / (gy[j + 1] - gy[j])
    return ((1 - tx) * (1 - ty) * V[i, j] + tx * (1 - ty) * V[i + 1, j]
            + (1 - tx) * ty * V[i, j + 1] + tx * ty * V[i + 1, j + 1])


@njit(cache=True)
def ref_mtpa_fw(iq_cmd, we, Vdc, Ld, Lq, psi, Imax):
    """MTPA below base speed, voltage-ellipse field weakening above it, current-circle limit.
    The voltage margin (0.9 Vdc/sqrt3) leaves headroom for the R drop and current dynamics."""
    iq = iq_cmd
    dL = Lq - Ld
    if dL > 0.0:
        a = psi / (2.0 * dL)
        i_d = a - np.sqrt(a * a + iq * iq)
        if i_d * i_d + iq * iq > Imax * Imax:   # MTPA point on the current circle
            i_d = (psi - np.sqrt(psi * psi + 8.0 * dL * dL * Imax * Imax)) / (4.0 * dL)
            q = np.sqrt(max(Imax * Imax - i_d * i_d, 0.0))
            iq = q if iq >= 0.0 else -q
    else:
        i_d = 0.0
    if abs(we) > 1.0:
        psi_lim = 0.9 * Vdc / SQ3 / abs(we)
        if (Ld * i_d + psi) ** 2 + (Lq * iq) ** 2 > psi_lim * psi_lim:
            disc = psi_lim * psi_lim - (Lq * iq) ** 2
            if disc > 0.0:
                i_fw = (-psi + np.sqrt(disc)) / Ld
                if i_fw < i_d:
                    i_d = i_fw
            else:
                i_d = -psi / Ld
                lim = psi_lim / Lq
                if iq > lim:
                    iq = lim
                elif iq < -lim:
                    iq = -lim
    if i_d < -Imax:
        i_d = -Imax
    if i_d * i_d + iq * iq > Imax * Imax:
        qmax = np.sqrt(max(Imax * Imax - i_d * i_d, 0.0))
        if iq > qmax:
            iq = qmax
        elif iq < -qmax:
            iq = -qmax
    return i_d, iq


@njit(cache=True)
def simulate(ctrl, pp, mp, cp, W1, b1, W2, b2, xm, xs, nnlo, nnhi,
             lut_x, lut_y, lut_v, iq_ref_ext, id_ref_ext, w_ref, TL_prof,
             speed_loop, mech_free, w0, iq0, seed):
    """pp: plant  [Rs, Ld, Lq, psi, p, J, B, Vdc, td, Esw_ref, Vref, Iref, V0, r_on]
       mp: model  [Rs, Ld, Lq, psi, J, B]  (nominal, used by controllers/observer)
       cp: control[Ts, nsub, KpD, KiD, KpQ, KiQ, Kpw, Kiw, Imax, lam_d, lam_q, lam_sw,
                   sig_i, sig_w, speed_div, log_div, l1, l2, Ilim_hard]
    """
    Rs, Ld, Lq, psi, p, J, B, Vdc, td = pp[0], pp[1], pp[2], pp[3], pp[4], pp[5], pp[6], pp[7], pp[8]
    Esw_ref, Vref, Iref, V0, r_on = pp[9], pp[10], pp[11], pp[12], pp[13]
    mRs, mLd, mLq, mpsi, mJ, mB = mp[0], mp[1], mp[2], mp[3], mp[4], mp[5]
    Ts = cp[0]
    nsub = int(cp[1])
    KpD, KiD, KpQ, KiQ, Kpw, Kiw, Imax = cp[2], cp[3], cp[4], cp[5], cp[6], cp[7], cp[8]
    lam_d_fix, lam_q_fix, lam_sw_fix = cp[9], cp[10], cp[11]
    sig_i, sig_w = cp[12], cp[13]
    speed_div = int(cp[14])
    log_div = int(cp[15])
    l1, l2, Ilim_hard = cp[16], cp[17], cp[18]
    use_mtpa = cp[19] > 0.5
    Iq_cmd_max = cp[20]
    h = Ts / nsub
    Kt_m = 1.5 * p * mpsi

    N = iq_ref_ext.shape[0]
    nfine = (N * nsub) // log_div
    log = np.zeros((N, NLOG))
    fine = np.zeros((nfine, NFINE))
    ncomm = np.zeros(3)
    E = np.zeros(5)  # [E_cu, E_cond, E_sw, E_mech_out, E_elec_in]

    np.random.seed(seed)

    # plant state
    i_d = 0.0
    i_q = iq0
    w = w0
    th = 0.0
    # inverter state
    gprev = np.zeros(3)      # last commanded gate per phase
    dtt = np.zeros(3)        # dead-time timers
    # command being applied in the current interval
    cmd_is_pwm = (ctrl == PI) or (ctrl == DB) or (ctrl == DPWM1) or (ctrl == DPWMI)
    duty = np.array([0.5, 0.5, 0.5])
    state_now = 0            # FCS state applied on [t_k, t_k+1)
    state_next = 0
    va_now = 0.0             # ideal alpha-beta voltage being applied now (for delay compensation)
    vb_now = 0.0
    va_next = 0.0
    vb_next = 0.0
    duty_next = np.array([0.5, 0.5, 0.5])
    # controller states
    intd = 0.0
    intq = mRs * iq0
    intw = iq0
    iq_ref = iq0
    w_hat = w0
    TL_hat = Kt_m * iq0 - mB * w0
    lam = np.zeros(2)
    xin4 = np.zeros(4)
    xin2 = np.zeros(2)
    fidx = 0

    for k in range(N):
        t = k * Ts
        # ---- measurement ----
        if not mech_free:
            w = w_ref[k]
        n1 = np.random.randn()
        n2 = np.random.randn()
        n3 = np.random.randn()
        id_m = i_d + sig_i * n1
        iq_m = i_q + sig_i * n2
        w_m = w + sig_w * 2.0 * np.pi / 60.0 * n3
        we_m = p * w_m
        # ---- load torque observer (same for all controllers) ----
        ew = w_m - w_hat
        w_hat += Ts * ((Kt_m * iq_m - TL_hat - mB * w_hat) / mJ + l1 * ew)
        TL_hat += -l2 * Ts * ew
        # ---- speed loop ----
        if speed_loop:
            if k % speed_div == 0:
                Tsw = Ts * speed_div
                e = w_ref[k] - w_m
                intw_new = intw + Kiw * Tsw * e
                u = Kpw * e + intw_new
                if u > Iq_cmd_max:
                    u = Iq_cmd_max
                elif u < -Iq_cmd_max:
                    u = -Iq_cmd_max
                else:
                    intw = intw_new
                iq_ref = u
        else:
            iq_ref = iq_ref_ext[k]
        id_ref = id_ref_ext[k]
        if use_mtpa:
            id_ref, iq_ref = ref_mtpa_fw(iq_ref, we_m, Vdc, mLd, mLq, mpsi, Imax)

        # ---- delay compensation: predict i(k+1) with the voltage applied now ----
        tha = th + 0.5 * we_m * Ts
        ca = np.cos(tha)
        sa = np.sin(tha)
        vd_a = va_now * ca + vb_now * sa
        vq_a = -va_now * sa + vb_now * ca
        id1 = id_m + Ts / mLd * (vd_a - mRs * id_m + we_m * mLq * iq_m)
        iq1 = iq_m + Ts / mLq * (vq_a - mRs * iq_m - we_m * mLd * id_m - we_m * mpsi)
        thb = th + 1.5 * we_m * Ts
        cb = np.cos(thb)
        sb = np.sin(thb)
        Vlim = Vdc / SQ3   # measured DC link = actual
        lam_d = lam_d_fix
        lam_q = lam_q_fix
        lam_sw = lam_sw_fix

        if cmd_is_pwm:
            if ctrl == PI or ctrl == DPWM1 or ctrl == DPWMI:
                ed = id_ref - id_m
                eq = iq_ref - iq_m
                vd = KpD * ed + intd - we_m * mLq * iq_m
                vq = KpQ * eq + intq + we_m * (mLd * id_m + mpsi)
                mag = np.sqrt(vd * vd + vq * vq)
                if mag > Vlim:
                    vd *= Vlim / mag
                    vq *= Vlim / mag
                else:
                    intd += KiD * Ts * ed
                    intq += KiQ * Ts * eq
            else:  # deadbeat on predicted i(k+1)
                vd = mRs * id1 + mLd * (id_ref - id1) / Ts - we_m * mLq * iq1
                vq = mRs * iq1 + mLq * (iq_ref - iq1) / Ts + we_m * (mLd * id1 + mpsi)
                mag = np.sqrt(vd * vd + vq * vq)
                if mag > Vlim:
                    vd *= Vlim / mag
                    vq *= Vlim / mag
            va_next = vd * cb - vq * sb
            vb_next = vd * sb + vq * cb
            # min-max injection SVPWM
            v1 = va_next
            v2 = -0.5 * va_next + 0.5 * SQ3 * vb_next
            v3 = -0.5 * va_next - 0.5 * SQ3 * vb_next
            vmax = max(v1, max(v2, v3))
            vmin = min(v1, min(v2, v3))
            vo = -0.5 * (vmax + vmin)
            if ctrl == DPWM1:
                # clamp the phase with the largest |v*| to the rail of its sign
                if vmax >= -vmin:
                    vo = 0.5 * Vdc - vmax
                else:
                    vo = -0.5 * Vdc - vmin
            elif ctrl == DPWMI:
                # candidates: max-voltage phase -> +rail, min-voltage phase -> -rail.
                # keep the one whose phase current (reference dq rotated to the PWM angle) is larger
                ial = id_ref * cb - iq_ref * sb
                ibe = id_ref * sb + iq_ref * cb
                j1 = ial
                j2 = -0.5 * ial + 0.5 * SQ3 * ibe
                j3 = -0.5 * ial - 0.5 * SQ3 * ibe
                if v1 == vmax:
                    jmax = j1
                elif v2 == vmax:
                    jmax = j2
                else:
                    jmax = j3
                if v1 == vmin:
                    jmin = j1
                elif v2 == vmin:
                    jmin = j2
                else:
                    jmin = j3
                if abs(jmax) >= abs(jmin):
                    vo = 0.5 * Vdc - vmax
                else:
                    vo = -0.5 * Vdc - vmin
            duty_next[0] = min(1.0, max(0.0, 0.5 + (v1 + vo) / Vdc))
            duty_next[1] = min(1.0, max(0.0, 0.5 + (v2 + vo) / Vdc))
            duty_next[2] = min(1.0, max(0.0, 0.5 + (v3 + vo) / Vdc))
        else:
            if ctrl == FCS_NN:
                xin4[0] = (id_ref - id_m - xm[0]) / xs[0]
                xin4[1] = (iq_ref - iq_m - xm[1]) / xs[1]
                xin4[2] = (w_m - xm[2]) / xs[2]
                xin4[3] = (TL_hat - xm[3]) / xs[3]
                nn_forward(xin4, W1, b1, W2, b2, lam)
                lam_d = nnlo + (nnhi - nnlo) * lam[0]
                lam_q = nnlo + (nnhi - nnlo) * lam[1]
            elif ctrl == FCS_SWNN:
                xin2[0] = (w_m - xm[0]) / xs[0]
                xin2[1] = (TL_hat - xm[1]) / xs[1]
                nn_forward(xin2, W1, b1, W2, b2, lam)
                lam_sw = 10.0 ** (nnlo + (nnhi - nnlo) * lam[0])
            elif ctrl == FCS_SWLUT:
                lam_sw = 10.0 ** lut_bilinear(lut_x, lut_y, lut_v, w_m, TL_hat)
            best = 1e300
            pa = (state_now >> 2) & 1
            pb = (state_now >> 1) & 1
            pc = state_now & 1
            for s in range(8):
                s_a = (s >> 2) & 1
                s_b = (s >> 1) & 1
                s_c = s & 1
                v_al = Vdc / 3.0 * (2.0 * s_a - s_b - s_c)
                v_be = Vdc / SQ3 * (s_b - s_c)
                vd = v_al * cb + v_be * sb
                vq = -v_al * sb + v_be * cb
                id2 = id1 + Ts / mLd * (vd - mRs * id1 + we_m * mLq * iq1)
                iq2 = iq1 + Ts / mLq * (vq - mRs * iq1 - we_m * mLd * id1 - we_m * mpsi)
                nsw = abs(s_a - pa) + abs(s_b - pb) + abs(s_c - pc)
                cost = (lam_d * (id_ref - id2) ** 2 + lam_q * (iq_ref - iq2) ** 2
                        + lam_sw * nsw + 1e-9 * nsw)
                if id2 * id2 + iq2 * iq2 > Ilim_hard * Ilim_hard:
                    cost += 1e6
                if cost < best:
                    best = cost
                    state_next = s
            s_a = (state_next >> 2) & 1
            s_b = (state_next >> 1) & 1
            s_c = state_next & 1
            va_next = Vdc / 3.0 * (2.0 * s_a - s_b - s_c)
            vb_next = Vdc / SQ3 * (s_b - s_c)

        # ---- log (controller view) ----
        log[k, LT] = t
        log[k, LIDREF] = id_ref
        log[k, LIQREF] = iq_ref
        log[k, LIDM] = id_m
        log[k, LIQM] = iq_m
        log[k, LWM] = w_m
        log[k, LWREF] = w_ref[k]
        log[k, LTLH] = TL_hat
        log[k, LLD] = lam_d
        log[k, LLQ] = lam_q
        log[k, LLSW] = lam_sw
        log[k, LSTATE] = state_now
        log[k, LTL] = TL_prof[k]

        # ---- plant over [t_k, t_k+1) with the command decided at k-1 ----
        TL = TL_prof[k]
        for j in range(nsub):
            tj = (j + 0.5) * h
            ce = np.cos(th)
            se = np.sin(th)
            ial = i_d * ce - i_q * se
            ibe = i_d * se + i_q * ce
            ia = ial
            ib = -0.5 * ial + 0.5 * SQ3 * ibe
            ic = -ia - ib
            iph0 = ia
            iph1 = ib
            iph2 = ic
            sv0 = 0.0
            sv1 = 0.0
            sv2 = 0.0
            for x in range(3):
                if cmd_is_pwm:
                    c = abs(2.0 * tj / Ts - 1.0)
                    g = 1.0 if c < duty[x] else 0.0
                else:
                    g = float((state_now >> (2 - x)) & 1)
                ix = iph0 if x == 0 else (iph1 if x == 1 else iph2)
                if g != gprev[x]:
                    ncomm[x] += 1.0
                    E[2] += Esw_ref * (Vdc / Vref) * (abs(ix) / Iref)
                    dtt[x] = td
                    gprev[x] = g
                if dtt[x] > 0.0:
                    sx = 1.0 if ix < 0.0 else 0.0
                    dtt[x] -= h
                else:
                    sx = g
                if x == 0:
                    sv0 = sx
                elif x == 1:
                    sv1 = sx
                else:
                    sv2 = sx
                E[1] += (V0 * abs(ix) + r_on * ix * ix) * h
            v_al = Vdc / 3.0 * (2.0 * sv0 - sv1 - sv2)
            v_be = Vdc / SQ3 * (sv1 - sv2)
            vd = v_al * ce + v_be * se
            vq = -v_al * se + v_be * ce
            we = p * w
            did = (vd - Rs * i_d + we * Lq * i_q) / Ld
            diq = (vq - Rs * i_q - we * Ld * i_d - we * psi) / Lq
            Te = 1.5 * p * (psi * i_q + (Ld - Lq) * i_d * i_q)
            E[0] += 1.5 * Rs * (i_d * i_d + i_q * i_q) * h
            E[3] += Te * w * h
            E[4] += 1.5 * (vd * i_d + vq * i_q) * h
            # stator flux-linkage derivative magnitude (stationary frame) for the iron-loss model
            ex = vd - Rs * i_d
            ey = vq - Rs * i_q
            dp2 = ex * ex + ey * ey
            rowi = (k * nsub + j) // log_div
            if rowi < nfine:
                dpm = np.sqrt(dp2)
                fine[rowi, FDP2] += dp2 / log_div
                fine[rowi, FDP15] += dpm * np.sqrt(dpm) / log_div
            if (k * nsub + j) % log_div == 0 and fidx < nfine:
                fine[fidx, FT] = t + j * h
                fine[fidx, FID] = i_d
                fine[fidx, FIQ] = i_q
                fine[fidx, FIA] = ia
                fine[fidx, FTE] = Te
                fine[fidx, FW] = w
                fidx += 1
            i_d += h * did
            i_q += h * diq
            if mech_free:
                w += h * (Te - TL - B * w) / J
            th += we * h
            if th > np.pi:
                th -= 2.0 * np.pi
        # ---- shift command pipeline (one-period delay) ----
        va_now = va_next
        vb_now = vb_next
        state_now = state_next
        duty[0] = duty_next[0]
        duty[1] = duty_next[1]
        duty[2] = duty_next[2]
        if not (np.isfinite(i_d) and np.isfinite(i_q) and np.isfinite(w)):
            return log[:k + 1], fine[:fidx], ncomm, E, False
    return log, fine, ncomm, E, True
