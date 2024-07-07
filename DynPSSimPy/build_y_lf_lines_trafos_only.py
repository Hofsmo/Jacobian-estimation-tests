import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import dynpssimpy.dynamic as dps
import dynpssimpy.solvers as dps_sol
import importlib
importlib.reload(dps)
import scipy.sparse as sp
import numpy as np


def build_y_lf_lines_trafos(ps):
    y_lf = np.zeros((ps.n_bus,) * 2, dtype=complex)
    for mdl in [ps.lines['Line'], ps.trafos['Trafo']]:  # ps.mdl_instructions['load_flow_adm']:
        data, (row_idx, col_idx) = mdl.load_flow_adm()
        sp_mat = sp.csr_matrix((
            data.flatten(),
            (row_idx.flatten(), col_idx.flatten())),
            shape=(ps.n_bus,) * 2
        )
        y_lf += sp_mat.todense()

    return y_lf
