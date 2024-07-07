import sys
from collections import defaultdict
import pandas as pd
import time
from sde_solver import EulerDAE_SDE
import numpy as np
    
def simulate_sys(ps, t_end, dt, fo_w, fo_a, load_a, load_t, gen_a,
                 hygov_model='HYGOV', verbose=True):
    
    # Get bus names
    bus_names = ps.buses['name']
    
    # Get state indices of states representing conductance (G) and susceptance (B) of loads:
    load_state_idx_g = ps.loads["DynamicLoadFiltered"].lpf_g.state_idx_global["x"]
    load_state_idx_b = ps.loads["DynamicLoadFiltered"].lpf_b.state_idx_global["x"]

    # Get generator speed state idx
    gen_speed_state_idx = ps.gen["GEN"].state_idx_global["speed"]
            

    # Solver
    n_loads = ps.loads["DynamicLoadFiltered"].n_units
    n_gen = ps.gen["GEN"].n_units
    dim_w = n_loads + n_gen
    sol = EulerDAE_SDE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0,
        ps.x_0,
        t_end,
        max_step=dt,
        dim_w=dim_w,
    )

    b_mat = np.zeros((len(sol.x), sol.dim_w))
    for i, (idx_g, idx_b) in enumerate(zip(load_state_idx_g, load_state_idx_b)):
        b_mat[idx_g, i] = load_a/load_t
    b_mat[idx_b, i] = load_a/load_t

    for i, idx in enumerate(gen_speed_state_idx):
        b_mat[idx, n_loads + i] = gen_a

    def b_func(t, x, v, b_mat=b_mat):
        return b_mat

    sol.b = b_func

    # Initialize simulation
    t = 0
    res = defaultdict(list)
    res["time"] = np.arange(0, t_end, dt)
    t_0 = time.time()

    for log_var in ["gen_angle", "gen_speed", "gen_power"]:
        res[log_var] = np.zeros((len(res["time"]), 4))  # There are 4 generators

    res["freq"] = np.zeros((len(res["time"]), 11))  # There are 11 buses

    # Run simulation
    for i in range(0, len(res["time"])):
        if verbose:
            sys.stdout.write("\r%d%%" % (t / (t_end) * 100))

        # Apply forcing signal
        ps.gov[hygov_model].set_input(
            input_name="aux_input",
            value=fo_a * np.sin(fo_w * t),
            idx=0,
        )

        # Simulate next step
        result = sol.step()
        x = sol.y
        v = sol.v
        dw = sol.dw  # random variable
        t = sol.t
        res["gen_angle"][i] = ps.gen["GEN"].angle(x, v).copy()
        res["gen_speed"][i] = ps.gen["GEN"].speed(x, v).copy()
        res["gen_power"][i] = ps.gen["GEN"].P_e(x, v)
        res["freq"][i] = ps.pll["PLL2"].freq_est(x, v)
        if hygov_model == 'HYGOV_mod':
            res['pll_freq_est'].append(ps.gov[hygov_model].pll.freq_est(x, v).copy())

    df = pd.concat(
        [
            pd.DataFrame(
                data=res["gen_angle"],
                index=res["time"],
                columns=["G1_angle", "G2_angle", "G3_angle", "G4_angle"],
            ),
            pd.DataFrame(
                data=res["gen_speed"],
                index=res["time"],
                columns=["G1_w", "G2_w", "G3_w", "G4_w"],
            ),
            pd.DataFrame(
                data=res["gen_power"],
                index=res["time"],
                columns=["G1_p", "G2_p", "G3_p", "G4_p"],
            ),
            pd.DataFrame(
                data=res["freq"],
                index=res["time"],
                columns=[bus_name+"_f" for bus_name in bus_names],
            ),
            pd.DataFrame(
                data=res["pll_freq_est"],
                index=res["time"],
                columns=["G1_f", "G2_f", "G3_f", "G4_f"],
            ),
        ],
        axis=1,
    )
    return df


