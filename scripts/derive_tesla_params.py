"""Derive the dq parameters used in configs/default.json from published MotorXP data
(Tesla Model 3 2018 rear motor, https://motorxp.com/wp-content/uploads/mxp_analysis_TeslaModel3.pdf).

Inputs taken from the whitepaper: Rs = 4.75 mOhm (20 C), p = 3, back-EMF 42 Vpk,LL/krpm,
peak point: 430 N m, 1000 Arms, advance angle 53.5 deg, 94.61 Vrms phase, 192 kW;
rotor lamination OD/ID 149.9/70 mm, stack 134 mm.
Assumptions: advance angle measured from the q axis; dq model with constant Ld, Lq at that point
(saturated values); voltage includes R drop; solid steel rotor for inertia.
"""
import numpy as np
from scipy.optimize import brentq

p = 3
psi = 42.0 / np.sqrt(3) / (1000 * 2 * np.pi / 60 * p)
I = 1000 * np.sqrt(2)
b = np.radians(53.5)
i_d, i_q = -I * np.sin(b), I * np.cos(b)
dL = (430 / (1.5 * p) - psi * i_q) / (i_d * i_q)          # Ld - Lq from the torque equation
we = p * 192e3 / 430
Vph, Rs = 94.61 * np.sqrt(2), 0.00475
f = lambda Ld: np.hypot(Rs * i_d - we * (Ld - dL) * i_q, Rs * i_q + we * (Ld * i_d + psi)) - Vph
Ld = brentq(f, 60e-6, 100e-6)
r1, r2, L = 0.1499 / 2, 0.035, 0.134
m = 7650 * np.pi * (r1 ** 2 - r2 ** 2) * L
print(f"psi_f = {psi:.6f} Wb")
print(f"Ld    = {Ld*1e6:.2f} uH, Lq = {(Ld-dL)*1e6:.2f} uH  (saturated, at the 430 N m point)")
print(f"corner speed = {192e3/430*60/(2*np.pi):.0f} rpm")
print(f"rotor inertia (solid estimate) = {0.5*m*(r1**2+r2**2):.4f} kg m^2")
