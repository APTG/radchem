import sys
import logging

import numpy as np
from scipy.integrate import odeint
# from scipy.integrate import solve_ivp

from model import RadChemModel

logger = logging.getLogger(__name__)


NA = 6.02214129e23      # Avogadro constant
EVJ = 6.24150934e18     # eV per joule


class Source():
    """
    Class for defining a beam source.
    """

    def __init__(self, dose=2.0, duration=10.0, plength=1e-6, pfreq=1e5):
        """
        dose : total dose to be delivered [Gy]
        duration : time interval where the dose will be delivered, always starting from t = 0 [s]
        plength : micropulse length [s]
        pfreq : micropulse frequency [Hz], set to "DC" for DC beam.
        """

        self.dose = dose
        self.duration = duration

        self.pfreq = pfreq

        # calculated from inital values
        self.doserate = dose / duration  # [Gy/sec]
        if pfreq == "DC":
            self.plength = duration
            self.pulsecount = 1
            self.beamon = duration
            self.beamoff = 0.0
            self.duty = 1.0
            self.pdoserate = self.doserate
            self.cyclelength = duration
        else:
            if plength > 1.0 / pfreq:
                logger.error("Pulse length cannot be larger than one period of pulse frequency.")
                exit()
            self.plength = plength
            self.pulsecount = int(duration * pfreq)  # total number of pulses given
            self.beamon = self.pulsecount * plength  # total time beam is ON
            self.beamoff = duration - self.beamon    # total time beam is OFF
            self.duty = self.beamon / duration       # duty cycle, 1.0 = DC beam
            self.pdoserate = dose / self.beamon      # peak micropulse rate
            self.cyclelength = 1 / self.pfreq

    def __str__(self):
        _str = ("# Dose                   : {:.3f} Gy\n".format(self.dose))
        _str += ("# Dose duration          : {:.3f} sec\n".format(self.duration))
        if self.pfreq == "DC":
            _str += ("# Micropulse frequency   : DC\n")
            _str += ("# Micropulse lenght      : DC\n")
            _str += ("# Cycle lenght           : DC\n")
        else:
            _str += ("# Micropulse frequency   : {:.3e} Hz\n".format(self.pfreq))
            _str += ("# Micropulse lenght      : {:.3e} sec\n".format(self.plength))
            _str += ("# Cycle lenght           : {:.3e} sec\n".format(self.cyclelength))

        _str += "\n"
        _str += ("# Average dose rate      : {:.3f} Gy/s\n".format(self.doserate))
        _str += ("# Micropulse dose rate   : {:.3f} Gy/s\n".format(self.pdoserate))
        _str += ("# Total number of pulses : {:.3e}\n".format(self.pulsecount))
        _str += ("# Actual beam ON time    : {:.3f} sec\n".format(self.beamon))
        _str += ("# Actual beam OFF time   : {:.3f} sec\n".format(self.beamoff))
        return _str

    def beam(self, t):
        """
        Calculate dose rate at time t, taking the given pulse structure into account.

        Returns : dose rate dD(t)/Dt at time t in Gy/sec, and zero if t < 0 sec
        """
        if t < 0.0:
            return 0.0

        if self.pfreq == "DC":
            return self.doserate

        # build a pulse
        npulse = int(t / self.cyclelength)  # current pulse number
        phase = (t - (npulse * self.cyclelength)) / self.cyclelength
        if phase < self.duty:
            return self.pdoserate
        else:
            return 0.0


# def dCdt(t, C, model, source=None):    # solve_ivp needs (t,C,...)
def dCdt(C, t, model, source=None):
    """
    Differential equation describing the change in concentration as a function of time.

    Input
    C_i(t) : given array of concentration of species i at time t [mol/liter]
    t      : time [sec]
    dDdt   : given doserate at time t [Gy/sec]

    Returns:
    dCi(t)/dt : change in concentration of all species i at time t. [mol/liter/sec]
    """
    dCdt = np.zeros(model.nspecies)
    # check first if any new ions are created due to irradiation
    if source:
        dDdt = source.beam(t)
        # print("# t, dDdt:", t, dDdt)
        if dDdt > 0.0:
            for k in range(model.nspecies):
                dCdt[k] = dDdt * model.gval[k] * 0.01 * EVJ / NA  # omitted times rho, assuming 1 kg = 1 liter

    # build rate constants for all species:
    for k in range(model.nspecies):
        # loop over each possible equation
        for j in range(model.neq):
            # in case the current species is involved:
            m = model.nmatrix[j][k]
            if m != 0:
                rate = m * model.rconst[j]
                # we still need to multiply with the concentrations of the constituents.
                # eq 1) AB -> A + B
                # eq 2) A + B -> AB
                # dC_AB / dt =  - 1 * rate_1 * C_AB
                #               + 1 * rate_2 * C_A * C_B
                for i in range(model.nspecies):
                    m = model.nmatrix[j][i]
                    if m == -1:
                        rate *= C[i]  # first order kinetics
                    elif m == -2:
                        rate *= C[i] * C[i]  # second order kinetics
                dCdt[k] += rate
    return dCdt


def print_species_header(model):
    """
    Provide header string for printing.
    """
    _str = "#"
    for i in range(model.nspecies):
        key = [key for key, value in model.symbol.items() if value == i][0]
        _str += " [{}]".format(key)
    print(_str)


def C(t, C0, model, source=None):
    """
    t      : array with time steps to calculate [sec]
    C_i(t) : array of concentrations of species i at time t [mol / l]
    C0     : array with starting condition for each species concentation [mol / l]
    source : beam source
    """
    # t_span = (min(t), max(t))
    sol = odeint(dCdt, C0, t, args=(model, source))
    # sol = solve_ivp(dCdt, t_span, y0=C0, args=(model, source), dense_output=True)
    return sol


# some start conditions
C0 = [  # ystart[NSPECIES] = [
    0,      # /* A0  : e-   */
    0,      # /* A1  : H    */
    0,      # /* A2  : OH   */
    0,      # /* A3  : H2O2 */
    4e-5,   # /* A4  : O2   */

    0,      # /* A5  : O2-  */
    0,      # /* A6  : HO2  */
    0,      # /* A7  : H2   */
    55.56,  # /* A8  : H2O  */
    0,      # /* A9  : OH-  */

    0,      # /* A10 : HO2- */
    0,      # /* A11 : H+   */
    0,      # /* A12 : O-   */
    0       # /* A13 : O3-  */
]


def main(args=sys.argv[1:]):
    """ Main function
    """

    # simulation parameters
    dose = 2.0          # [Gy]
    pulsewidth = 0.01   # [sec]
    freq = 13.0         # [Hz]

    # simulation interval
    t_start = -1.0      # simulation start time, beam will only be on at t >= 0 [sec]
    t_stop = 2.0        # stop at this time. Beam only be on at t >= 0 [sec]
    steps = 1001

    model = RadChemModel

    # s = Source(dose, t_stop, pulsewidth, "DC")
    s = Source(dose, t_stop, pulsewidth, freq)
    print(s)

    t = np.linspace(t_start, t_stop, steps)
    result = C(t, C0, RadChemModel, s)

    # sol = solve_ivp(dCdt, (t_start, t_stop), y0=C0, args=(RadChemModel, s), dense_output=True)
    # result = sol.sol(t)

    # print data for certain species:
    symb = model.symbol
    a = result[:, symb["H+"]]
    b = result[:, symb["OH-"]]
    c = result[:, symb["e-"]]
    d = result[:, symb["OH"]]

    for i, tt in enumerate(t):
        # print(tt, s.beam(tt), a[i], b[i], c[i], d[i])
        _str = "{:.3f}".format(tt)
        _str += " {:.3f}".format(s.beam(tt))
        _str += " {:.3e}".format(a[i])
        _str += " {:.3e}".format(b[i])
        _str += " {:.3e}".format(c[i])
        _str += " {:.3e}".format(d[i])
        print(_str)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
