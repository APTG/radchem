import argparse
import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from typing import Optional, List, Sequence

import matplotlib.pylab as plt  # type: ignore
import numpy as np  # type: ignore
from scipy.integrate import solve_ivp  # type: ignore
from tqdm import tqdm  # type: ignore

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

    def beam_edges(self) -> Sequence[float]:
        leading_edges = np.arange(start=0.0, stop=self.duration, step=self.cyclelength)
        trailing_edges = np.arange(start=self.plength, stop=self.duration, step=self.cyclelength)
        return np.column_stack([leading_edges, trailing_edges]).flatten()

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


def dCdt(C: Sequence[float], t: float, model: RadChemModel, source: Optional[Source] = None, progress_bar=None) -> Sequence[float]:
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
    progress_bar.update()  # update progress bar

    dCdt_vec = model.dCdt_f(C, t)
    # check first if any new ions are created due to irradiation
    if source:
        dDdt = source.beam(t)
        # print("# t, dDdt:", t, dDdt)
        if dDdt > 0.0:
            for k in range(len(dCdt_vec)):
                dCdt_vec[k] = dDdt * model.gval[k] * 0.01 * EVJ / NA  # omitted times rho, assuming 1 kg = 1 liter

    return dCdt_vec


def dCdt2(t: float, C: Sequence[float], model: RadChemModel, source: Optional[Source] = None, progress_bar=None) -> Sequence[float]:
    return dCdt(C, t, model, source, progress_bar)


def dCdt_Jac(C: Sequence[float], t: float, model: RadChemModel, source: Optional[Source] = None, progress_bar=None) -> Sequence[float]:
    dCdt_jac_res = model.dCdt_Jac_f(C, t, model, source)
    return dCdt_jac_res


def dCdt_Jac2(t: float, C: Sequence[float], model: RadChemModel, source: Optional[Source] = None, progress_bar=None) -> Sequence[float]:
    return dCdt_Jac(C, t, model, source, progress_bar)


def C(t: Sequence[float], C0: Sequence[float], model: RadChemModel, source: Optional[Source] = None) -> List[float]:
    """
    t      : array with time steps to calculate [sec]
    C_i(t) : array of concentrations of species i at time t [mol / l]
    C0     : array with starting condition for each species concentration [mol / l]
    source : beam source
    """

    # new API:
    max_step = 1e-3
    if source:
        max_step = source.plength / 10.0  # TODO: investigate other step limits

    expected_no_of_steps = 2 * int((max(t) - min(t)) / max_step)

    with tqdm(total=expected_no_of_steps) as progress_bar:
        sol = solve_ivp(fun=dCdt2,
                        t_span=(min(t), max(t)),
                        y0=C0,
                        method='BDF',
                        t_eval=t,
                        args=(model, source, progress_bar),
                        jac=dCdt_Jac2,  # TODO: investigate disabling it
                        rtol=1e-6,  # TODO: investigate default tolerance
                        max_step=max_step,
                        #                   vectorized=True,  # TODO investigate parallel version
                        )
    return sol.y


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

sim_output_filename = 'results.csv'


def run():
    start = time.time()

    # simulation parameters
    dose = 10.0  # [Gy]
    pulsewidth = 2e-6  # [sec]
    freq = 100.0  # [Hz]

    # simulation interval
    t_start = -1e-6  # simulation start time, beam will only be on at t >= 0 [sec]
    t_stop = 25 * 1e-3  # stop at this time. Beam only be on at t >= 0 [sec]
    steps = 2500

    model = RadChemModel

    # s = Source(dose, t_stop, pulsewidth, "DC")
    s = Source(dose, t_stop, pulsewidth, freq)
    print(s)

    t = np.linspace(t_start, t_stop, steps)

    logger.info("Initial concentration")
    for i, c0_item in enumerate(C0):
        if C0[i] > 0:
            logger.info("\tC[{} ({})] = {}".format(i, model.species_symbols[i], C0[i]))

    # run simulation
    result = C(t, C0, RadChemModel, s)

    # simulate beam profile
    beam_profile = np.zeros_like(t)
    for i, beam_time in enumerate(t):
        beam_profile[i] = s.beam(beam_time)

    # save simulation output for later processing (i.e. plotting)
    np.savetxt(fname=sim_output_filename,
               X=np.column_stack([t, beam_profile, result.T]),
               header="t beam " + " ".join([str(s) for s in RadChemModel.species_symbols]))
    print("Saved simulation output into {}".format(sim_output_filename))
    end = time.time()
    print("Simulating took {:3.3f} sec".format(end - start))
    return 0


def plot():
    start = time.time()
    data = np.genfromtxt(sim_output_filename, delimiter=' ', names=True, dtype=None)
    fig, ax = plt.subplots(figsize=(12, 10))
    for i in range(1, len(data.dtype)):
        ydata = data[data.dtype.names[i]]
        if ydata.max() > 1e-10:
            ax.plot(data['t'], ydata, '.', label=data.dtype.names[i], markersize=1)
    ax.set_yscale('log')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Concentration [mol/liter]")
    ax.grid()
    ax.legend(loc=0)
    # fig.savefig("plot.pdf")
    fig.savefig("plot.png")
    ax.set_ylim(1e-20, 1e2)
    # fig.savefig("plot_zoom1.pdf")
    fig.savefig("plot_zoom1.png")
    ax.set_ylim(1e-10, 1e0)
    # fig.savefig("plot_zoom2.pdf")
    fig.savefig("plot_zoom2.png")

    end = time.time()
    print("Plotting took {:3.3f} sec".format(end - start))
    return 0


def main(args=sys.argv[1:]):
    """ Main function
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbosity', action='count', help="increase output verbosity", default=0)
    parser.add_argument('-r', '--run', help="run simulation and save output", action='store_true')
    parser.add_argument('-p', '--plot', help="plot results", action='store_true')
    parsed_args = parser.parse_args(args)

    if parsed_args.verbosity == 1:
        logger.setLevel(logging.INFO)
    elif parsed_args.verbosity > 1:
        logger.setLevel(logging.DEBUG)

    if parsed_args.run:
        status = run()
        if status != 0:
            return status

    if parsed_args.plot:
        status = plot()
        if status != 0:
            return status

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
