from tops.dyn_models.utils import DAEModel, auto_init
from tops.dyn_models.blocks import *
from tops.dyn_models.gov import GOV
from tops.dyn_models.pll import PLL2


class TGOV1(DAEModel, GOV):
    """This is similar to the TGOV1 model in DynPSSimPy, but with an auxiliary input signal"""

    def input_list(self):
        return ["input", "P_n_gen", "aux_input"]

    def add_blocks(self):
        p = self.par
        self.droop = Gain(K=1 / p["R"])
        self.time_constant_lim = TimeConstantLims(
            T=p["T_1"], V_min=p["V_min"], V_max=p["V_max"]
        )
        self.lead_lag = LeadLag(T_1=p["T_2"], T_2=p["T_3"])
        self.damping_gain = Gain(K=p["D_t"])

        self.droop.input = (
            lambda x, v: -self.input(x, v) + self.int_par["bias"] + self.aux_input(x, v)
        )
        self.time_constant_lim.input = lambda x, v: self.droop.output(x, v)
        self.lead_lag.input = lambda x, v: self.time_constant_lim.output(x, v)
        self.damping_gain.input = lambda x, v: self.input(x, v)

        self.output = lambda x, v: self.lead_lag.output(
            x, v
        ) - self.damping_gain.output(x, v)

    def int_par_list(self):
        return ["bias"]

    def init_from_connections(self, x0, v0, output_0):
        # auto_init(self, x0, v0, output_0['output'])
        p = self.par
        self.int_par["bias"] = self.droop.initialize(
            x0,
            v0,
            self.time_constant_lim.initialize(
                x0, v0, self.lead_lag.initialize(x0, v0, output_0["output"])
            ),
        )


class HYGOV(GOV, DAEModel):
    """
    Implementation of the HYGOV model. Some limiters are missing.
    This is similar to the TGOV1 model in DynPSSimPy, but with an auxiliary input signal.
    """

    def input_list(self):
        return ["input", "P_n_gen", "aux_input"]

    def int_par_list(self):
        return ["bias"]

    def add_blocks(self):
        p = self.par
        self.time_constant_1 = TimeConstant(T=p["T_f"])
        self.pi_reg = PIRegulator2(
            T_1=p["T_r"], T_2=p["T_r"] * p["r"]
        )  # This should have limits!
        self.gain = Gain(K=p["R"])
        self.time_constant_2 = TimeConstant(T=p["T_g"])
        self.gain_A_t = Gain(K=p["A_t"])
        self.integrator = Integrator2(T=p["T_w"])

        self.time_constant_1.input = (
            lambda x, v: -self.input(x, v)
            + self.int_par["bias"]
            - p["R"] * self.c(x, v)
            + self.aux_input(x, v)
        )
        self.pi_reg.input = self.time_constant_1.output  # This should have a limiter
        self.c = self.pi_reg.output
        self.time_constant_2.input = self.c
        self.g = self.time_constant_2.output
        self.q = self.integrator.output
        self.div = lambda x, v: self.q(x, v) / (
            self.time_constant_2.output(x, v)
        )  #  + 1e-6)  # Quick-fix to avoid division by zero.
        self.h = lambda x, v: self.div(x, v) ** 2
        self.integrator.input = lambda x, v: -self.h(x, v) + 1
        self.gain_A_t.input = lambda x, v: (self.q(x, v) - p["q_nl"]) * self.h(x, v)
        self.output = self.gain_A_t.output

    def init_from_connections(self, x0, v0, output_0):
        auto_init(self, x0, v0, output_0["output"])

class HYGOV_mod(HYGOV):
    """
    Implementation of the HYGOV model. Some limiters are missing.
    This is similar to the TGOV1 model in DynPSSimPy, but with an auxiliary input signal, and with frequency estimator.
    """

    def add_blocks(self):
        p = self.par
        
        super().add_blocks()
        self.pll = PLL2(K_p=self.par['PLL_K_p'], K_i=self.par['PLL_K_i'], bus=p['bus'], sys_par=self.sys_par)
        self.time_constant_1.input = (
            lambda x, v: -self.pll.freq_est(x, v)  # self.input(x, v)
            + self.int_par["bias"]
            - p["R"] * self.c(x, v)
            + self.aux_input(x, v)
        )

    def init_from_connections(self, x0, v0, output_0):
        auto_init(self, x0, v0, output_0["output"])
        
        # Make sure PLL is correctly initialized (might initialize in unstable operating point from iterative solution)
        output_value = np.angle(v0[self.pll.bus_idx_red['terminal']])
        self.pll.integrator.initialize(x0, v0, output_value)
        
