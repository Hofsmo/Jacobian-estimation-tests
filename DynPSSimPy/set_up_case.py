import tops.dynamic as dps
import user_lib
from utils import remove_model_data

def create_ps(model, load_t, hygov_model='HYGOV'):
    # Power system model
    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)

    # Replace const. impedance load model with DynamicLoadFiltered:
    ps.add_model_data(
        {
            "loads": {
                "DynamicLoadFiltered": [
                    [*ps.loads["Load"].par.dtype.names, "T_g", "T_b"],
                    *[[*load_data, load_t, load_t] for load_data in ps.loads["Load"].par],
                ]
            }
        }
    )
    
    # Get bus names
    bus_names = ps.buses['name']
    
    ps.add_model_data({'pll': {
        'PLL2': [
             ['name',        'K_p',  'K_i',  'bus'   ],
            *[[f'PLL{i}',    100,     100,      bus_name  ] for i, bus_name in enumerate(bus_names)],
        ]
    }})

    ps.add_model_data(
        {
            "gov": {
                hygov_model: [
                    [
                        "name",
                        "gen",
                        "R",
                        "r",
                        "T_f",
                        "T_r",
                        "T_g",
                        "A_t",
                        "T_w",
                        "q_nl",
                        "D_turb",
                        "G_min",
                        "V_elm",
                        "G_max",
                        "P_N",
                        "PLL_K_p",
                        "PLL_K_i",
                        "bus"
                    ],
                    *[
                        [
                            f"GOV{i}",
                            gen["name"],
                            0.06,
                            0.4,
                            0.1,
                            7.5,
                            0.5,
                            1,
                            1,
                            0.01,
                            0.0,
                            0,
                            0.15,
                            1,
                            0,
                            10,
                            100,
                            gen["bus"]
                        ]
                        for i, gen in enumerate(ps.gen["GEN"].par)
                    ],
                ]
            }
        }
    )

    ps.gen["GEN"].par["D"] = 0.01

    remove_model_data(ps, "loads", "Load")
    remove_model_data(ps, "gov", "TGOV1")

    ps.init_dyn_sim()
    return ps

