import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import dynpssimpy.dynamic as dps
import dynpssimpy.solvers as dps_sol
import importlib
import numpy as np
try:
    from DynPSSimPy.utils import remove_model_data
    import DynPSSimPy.user_lib as user_lib
except:
    from utils import remove_model_data
    import user_lib

if __name__ == '__main__':

    # Load model
    import dynpssimpy.ps_models.k2a as model_data
    importlib.reload(model_data)
    model = model_data.load()

    # Power system model
    

    # hygov_model = 'HYGOV_mod'
    res_store = []
    for hygov_model in ['HYGOV', 'HYGOV_mod']:
        ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)
        ps.add_model_data({"gov": {hygov_model: [
            ["name", "gen", "R", "r", "T_f", "T_r", "T_g", "A_t", "T_w", "q_nl", "D_turb", "G_min", "V_elm", "G_max", "P_N", 'PLL_K_p', 'PLL_K_i', 'bus'],
            *[[f"GOV{i}", gen["name"], 0.06, 0.4, 0.1, 7.5, 0.5, 1, 1, 0.01, 0.01, 0, 0.15, 1, 0, 10, 100, gen['bus']] for i, gen in enumerate(ps.gen["GEN"].par)],
            ]}}
        )

        remove_model_data(ps, "gov", "TGOV1")

        ps.init_dyn_sim()
        print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))

        t_end = 20
        x_0 = ps.x_0.copy()

        # Solver
        sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, t_end, max_step=5e-3)

        # Initialize simulation
        t = 0
        res = defaultdict(list)
        t_0 = time.time()

        sc_bus_idx = ps.gen['GEN'].bus_idx_red['terminal'][0]

        # Run simulation
        event_flag = True
        while t < t_end:
            sys.stdout.write("\r%d%%" % (t/(t_end)*100))

            # Short circuit
            if t > 1 and event_flag:
                event_flag = False
                ps.lines['Line'].event(ps, ps.lines['Line'].par['name'][2], 'disconnect')

            # Simulate next step
            result = sol.step()
            x = sol.y
            v = sol.v
            t = sol.t

            dx = ps.ode_fun(0, ps.x_0)

            # Store result
            res['t'].append(t)
            res['gen_speed'].append(ps.gen['GEN'].speed(x, v).copy())
            res['gen_angle'].append(ps.gen['GEN'].angle(x, v).copy())
            res['gen_v_angle'].append(np.angle(ps.gen['GEN'].v_t(x, v)).copy())

            if hygov_model == 'HYGOV_mod':
                res['pll_freq_est'].append(ps.gov[hygov_model].pll.freq_est(x, v).copy())
                res['pll_angle_est'].append(ps.gov[hygov_model].pll.output(x, v).copy())

        print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))

        res_store.append(res)

    plt.figure()
    for res, linestyle in zip(res_store, ['-', '--']):
        for i in range(ps.gen['GEN'].n_units):
            plt.plot(res['t'], np.array(res['gen_speed'])[:, i], color=f'C{i}', linestyle=linestyle, label=f'Speed G{i}')
            
            # if hygov_model == 'HYGOV_mod':
                # plt.plot(res['t'], np.array(res['pll_freq_est'])[:, i], color=f'C{i}', linestyle='--', label=f'Freq est G{i}')
        plt.xlabel('Time [s]')
        plt.ylabel('Gen. speed, estimated frequency')
        plt.legend()
        # plt.show()

    plt.figure()
    for res, linestyle in zip(res_store, ['-', '--']):
        for i in range(ps.gen['GEN'].n_units):
            # plt.plot(res['t'], np.array(res['gen_angle'])[:, i], color=f'C{i}')
            plt.plot(res['t'], np.array(res['gen_v_angle'])[:, i], color=f'C{i}', linestyle=linestyle, label=f'Terminal voltage angle B{i}')
            
            # if hygov_model == 'HYGOV_mod':
                # plt.plot(res['t'], np.array(res['pll_angle_est'])[:, i], color=f'C{i}', linestyle='--', label=f'Estimated voltage angle B{i}')
        plt.xlabel('Time [s]')
        plt.ylabel('Gen. angle, estimated bus voltage angle')
        # plt.show()
    plt.show()