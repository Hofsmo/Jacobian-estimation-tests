# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
from multiprocessing import Pool
import pandas as pd
import numpy as np
from generate_data import simulate_sys
# Load model
import dynpssimpy.ps_models.k2a as model_data
import dynpssimpy.dynamic as dps
from set_up_case import create_ps

from sparculing.helper_functions import get_lin_sys

from scipy.signal import decimate

import matplotlib.pyplot as plt
# %matplotlib widget
# -

model = model_data.load()

# +
# Case parameters
t_end = 500
dt = 5e-3
load_a = 0.1
# Forced oscillations
fo_a = 0  # 0.005
fo_w = 2 * np.pi * 0.2
    
# Noise
gen_a = 0
# Load
load_t = 1e1
ps = create_ps(model, load_t)
df = simulate_sys(ps, t_end, dt, fo_w, fo_a, load_a, load_t, gen_a)
# -

df["G1_w"].plot()

df.columns

df["G1_angle"].plot()

angles = df[["G1_angle", "G2_angle", "G3_angle", "G4_angle"]].to_numpy()

omegas = df[["G1_w", "G2_w", "G3_w", "G4_w"]].to_numpy()

M = 2*np.array([6.5, 6.5, 6.175, 6.175])
D = np.array([0.01, 0.01, 0.01, 0.01])

CC = np.cov(np.concatenate((np.transpose(angles), np.transpose(omegas))))

Cdd = CC[0:4, 0:4]

Cww = CC[4:8, 4:8]

Cwd = CC[4:8, 0:4]

J = np.diag(M)*Cww*np.linalg.inv(Cdd)-np.diag(D)*Cwd*np.linalg.inv(Cdd)

dP_dd = np.array(Cww)*np.linalg.inv(Cdd)

1/M-1/(2*6.5)

A = np.concatenate(
    (
        np.concatenate((np.zeros((4, 4)), np.identity(4)), axis=1),
        np.concatenate((-np.diag(1/M)*J, -np.diag(1/M)*np.diag(D)), axis=1)),axis=0)

lin_A=np.linalg.eig(A)

lin_sys = get_lin_sys(ps)

plt.figure()
plt.scatter(lin_A.eigenvalues.real, lin_A.eigenvalues.imag)
plt.scatter(lin_sys.eigs.real, lin_sys.eigs.imag)

np.min(lin_sys.damping)

np.min(-lin_A.eigenvalues.real/np.absolute(lin_A.eigenvalues))
