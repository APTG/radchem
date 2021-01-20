import argparse
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional, List, Sequence

import numpy as np  # type: ignore
from scipy.integrate import odeint  # type: ignore

from model import RadChemModel

NA = 6.02214129e23  # Avogadro constant
EVJ = 6.24150934e18  # eV per joule

# add a rotating handler
log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s')
handler = RotatingFileHandler("test.log", mode='w', maxBytes=1024 * 1000, backupCount=1)
handler.setFormatter(log_formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
try:
    os.remove("test.log")
except FileNotFoundError:
    pass
logger.addHandler(handler)


class Source:
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
            self.beamoff = duration - self.beamon  # total time beam is OFF
            self.duty = self.beamon / duration  # duty cycle, 1.0 = DC beam
            self.pdoserate = dose / self.beamon  # peak micropulse rate
            self.cyclelength = 1 / self.pfreq

    def __str__(self) -> str:
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

    def beam(self, t: float) -> float:
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


def dCdt(C: Sequence[float], t: float, model: RadChemModel, source: Optional[Source] = None) -> Sequence[float]:
    """
    Differential equation describing the change in concentration as a function of time.

    Input
    C_i(t) : given array of concentration of species i at time t [mol/liter]
    t      : time [sec]
    dDdt   : given doserate at time t [Gy/sec]

    Returns:
    dCi(t)/dt : change in concentration of all species i at time t. [mol/liter/sec]
    """
    logger.info("t = {}".format(t))

    dCdt_vec = model.dCdt_f(C, t)
    # check first if any new ions are created due to irradiation
    if source:
        dDdt = source.beam(t)
        # print("# t, dDdt:", t, dDdt)
        if dDdt > 0.0:
            for k in range(len(dCdt_vec)):
                dCdt_vec[k] = dDdt * model.gval[k] * 0.01 * EVJ / NA  # omitted times rho, assuming 1 kg = 1 liter

    return dCdt_vec


def dCdt_Jac(C: Sequence[float], t: float, model: RadChemModel, source: Optional[Source] = None) -> Sequence[float]:
    dCdt_jac_res = model.dCdt_Jac_f(C, t)
    return dCdt_jac_res


def C(t: float, C0: Sequence[float], model: RadChemModel, source: Optional[Source] = None) -> List[float]:
    """
    t      : array with time steps to calculate [sec]
    C_i(t) : array of concentrations of species i at time t [mol / l]
    C0     : array with starting condition for each species concentration [mol / l]
    source : beam source
    """
    sol, info_dict = odeint(dCdt, C0, t, args=(model, source), full_output=True, tcrit=np.linspace(1e-5, 2, 1000))
    print(info_dict)
    return sol


# some start conditions
C0 = [  # ystart[NSPECIES] = [
    0,  # /* A0  : e-   */
    0,  # /* A1  : H    */
    0,  # /* A2  : OH   */
    0,  # /* A3  : H2O2 */
    4e-5,  # /* A4  : O2   */

    0,  # /* A5  : O2-  */
    0,  # /* A6  : HO2  */
    0,  # /* A7  : H2   */
    55.56,  # /* A8  : H2O  */
    0,  # /* A9  : OH-  */

    0,  # /* A10 : HO2- */
    0,  # /* A11 : H+   */
    0,  # /* A12 : O-   */
    0  # /* A13 : O3-  */
]


def main(args=sys.argv[1:]):
    """ Main function
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbosity', action='count', help="increase output verbosity", default=0)
    args = parser.parse_args(args)

    if args.verbosity == 1:
        logger.setLevel(logging.INFO)
    elif args.verbosity > 1:
        logger.setLevel(logging.DEBUG)

    # simulation parameters
    dose = 2000000.0  # [Gy]
    pulsewidth = 0.01  # [sec]
    freq = 10.0  # [Hz]

    # simulation interval
    t_start = -1.0  # simulation start time, beam will only be on at t >= 0 [sec]
    t_stop = 2.0  # stop at this time. Beam only be on at t >= 0 [sec]
    steps = 3000

    model = RadChemModel

    # s = Source(dose, t_stop, pulsewidth, "DC")
    s = Source(dose, t_stop, pulsewidth, freq)
    print(s)

    t = np.linspace(t_start, t_stop, steps)

    logger.info("Initial concentration")
    for i, c0_item in enumerate(C0):
        if C0[i] > 0:
            logger.info("\tC[{} ({})] = {}".format(i, model.species_symbols[i], C0[i]))

    result = C(t, C0, RadChemModel, s)
    print(result.shape)
    np.savetxt("result.txt", result)

    return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
