from functools import reduce, lru_cache
from operator import mul
from typing import Callable, Dict, List, Tuple, Any, Sequence

import sympy as sym  # type: ignore


def prod(seq):
    """multiply all items from sequence"""
    return reduce(mul, seq) if seq else 1


class RadChemModel:
    # time symbol
    t = sym.symbols('t', real=True)

    # species symbols
    species_symbols = sym.symbols(
        ['e^{-}', 'H', 'OH', 'H_{2}O_{2}', 'O_2', 'O_2^{-}', 'HO2', 'H2', 'H_{2}O', 'OH^{-}', 'HO_2^{-}', 'H^{+}',
         'O^{-}', 'O_3^{-}'], real=True, nonnegative=True)
    em, H, OH, H2O2, O2, O2m, HO2, H2, H2O, OHm, HO2m, Hp, Om, O3m = species_symbols

    @property
    def nspecies(self):
        return len(self.species_symbols)

    # reaction constants
    rconst_values = (
        # /* v1-v5*/
        6.44e9,
        2.64e10,
        3.02e10,
        1.41e10,
        1.79e10,

        # /* v6-v10*/
        1.30e10,
        1.28e10,
        5.43e9,
        1.53e10,
        5.16e7,

        # /* v11-v15*/
        1.32e10,
        9.98e9,
        9.98e9,
        4.74e9,
        4.15e7,

        # /* v16-v20*/
        2.87e7,
        1.08e10,
        1.10e10,
        6.64e5,
        7.58e7,

        # /* v21-v25*/
        1.95e-5,
        1.10e11,
        7.86e-2,
        4.78e10,
        1.27e10,

        # /* v26-v30*/
        1.36e6,
        6.32e0,
        2.25e10,
        1.55e1,
        2.49e7,

        # /* v31-v35*/
        7.86e-2,
        4.78e10,
        1.27e10,
        1.36e6,
        7.14e5,

        # /* v36-v40*/
        4.78e10,
        1.27e10,
        1.36e6,
        1.21e8,
        5.53e8,

        # /* v41-v45*/
        8.29e9,
        7.60e9,
        3.50e9,
        2.31e10,
        3.70e9,

        # /* v46-v50*/
        2.68e3,
        4.00e8,
        6.00e8,
        5.00e-1,
        1.30e-1,

        # /* dummy value for electron creation, */
        # /* is is going to be time dependent, and overridden by program. */
        0.00e0
    )

    rconst_symbols = sym.symbols(['k_{:02}'.format(i) for i in range(len(rconst_values))], real=True, nonnegative=True)

    # /* G-values at 25 deg C [#/100eV] */
    gval = (  # gval[NSPECIES] = (
        2.645,  # /* A0  : e-   */
        0.572,  # /* A1  : H    */
        2.819,  # /* A2  : OH   */
        0.646,  # /* A3  : H2O2 */
        0,  # /* A4  : O2   */

        0,  # /* A5  : O2-  */
        0,  # /* A6  : HO2  */
        0.447,  # /* A7  : H2   */
        -4.541,  # /* A8  : H2O  */
        0.430,  # /* A9  : OH-  */

        0,  # /* A10 : HO2- */
        3.075,  # /* A11 : H+   */
        0.430,  # /* A12 : O-   */
        0  # /* A13 : O3-  */
    )

    reactions: List[Tuple[Any, Dict[Any, int], Dict[Any, int], str]] = [
        # (coeff, r_stoich, net_stoich, equation)
        # # /* v1-v5*/
        # # /*0           4     5           9    10        14*/
        # (-2, 0, 0, 0, 0,    0, 0, 1, 0, 2,    0, 0, 0, 0),   # /* e-  + e-   -> H2  + 2OH- */
        # (-1, -1, 0, 0, 0,    0, 0, 1, 0, 1,    0, 0, 0, 0),  # /* e-  + H    -> H2  + OH-  */
        # (-1, 0, -1, 0, 0,    0, 0, 0, 0, 1,    0, 0, 0, 0),  # /* e-  + OH   -> OH-        */
        # (-1, 0, 1, -1, 0,    0, 0, 0, 0, 1,    0, 0, 0, 0),  # /* e-  + H2O2 -> OH- + OH   */
        # (-1, 0, 0, 0, -1,    1, 0, 0, 0, 0,    0, 0, 0, 0),  # /* e-  + O2   -> O2-        */
        (rconst_symbols[0], {em: 2}, {em: -2, H2: 1, OHm: 2}, "e- + e- -> H2 + 2OH-"),
        (rconst_symbols[1], {em: 1, H: 1}, {em: -1, H: -1, H2: 1, OHm: 1}, "e-  + H    -> H2  + OH-"),
        (rconst_symbols[2], {em: 1, OH: 1}, {em: -1, OH: -1, OHm: 1}, "e-  + OH   -> OH-"),
        (rconst_symbols[3], {em: 1, H2O2: 1}, {em: -1, H2O2: -1, OHm: 1, OH: 1}, "e-  + H2O2 -> OH- + OH"),
        (rconst_symbols[4], {em: 1, O2: 1}, {em: -1, O2: -1, O2m: 1}, "e-  + O2   -> O2-"),

        # # /* v5-v10*/
        # (-1, 0, 0, 0, 0,   -1, 0, 0, 0, 1,    1, 0, 0, 0),   # /* e-  + O2-  -> HO2- + OH- */
        # (-1, 0, 0, 0, 0,    0, -1, 0, 0, 0,    1, 0, 0, 0),  # /* e-  + HO2  -> HO2-       */
        # (0, -2, 0, 0, 0,    0, 0, 1, 0, 0,    0, 0, 0, 0),   # /* 2H         -> H2         */
        # (0, -1, -1, 0, 0,    0, 0, 0, 1, 0,    0, 0, 0, 0),  # /* H   + OH   -> H2O        */
        # (0, -1, 1, -1, 0,    0, 0, 0, 1, 0,    0, 0, 0, 0),  # /* H   + H2O2 -> OH   + H2O */
        (rconst_symbols[5], {em: 1, O2m: 1}, {em: -1, O2m: -1, HO2m: 1, OHm: 1}, "e-  + O2-  -> HO2- + OH-"),
        (rconst_symbols[6], {em: 1, HO2: 1}, {em: -1, HO2: -1, HO2m: 1}, "e-  + HO2  -> HO2-"),
        (rconst_symbols[7], {H: 2}, {H: -2, H2: 1}, "2H         -> H2"),
        (rconst_symbols[8], {H: 1, OH: 1}, {H: -1, OH: -1, H2O: 1}, "H   + OH   -> H2O"),
        (rconst_symbols[9], {H: 1, H2O2: 1}, {H: -1, H2O2: -1, OH: 1, H2O: 1}, "H   + H2O2 -> OH   + H2O"),

        # # /* v11-v15*/
        # (0, -1, 0, 0, -1,    0, 1, 0, 0, 0,    0, 0, 0, 0),  # /* H   + O2   -> HO2        */
        # (0, -1, 0, 1, 0,    0, -1, 0, 0, 0,    0, 0, 0, 0),  # /* H   + HO2  -> H2O2       */
        # (0, -1, 0, 0, 0,   -1, 0, 0, 0, 0,    1, 0, 0, 0),   # /* H   + O2-  -> HO2-       */
        # (0, 0, -2, 1, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0),   # /* OH  + OH   -> H2O2       */
        # (0, 1, -1, 0, 0,    0, 0, -1, 1, 0,    0, 0, 0, 0),  # /* OH  + H2   -> H    + H2O */
        (rconst_symbols[10], {H: 1, O2: 1}, {H: -1, O2: -1, HO2: 1}, "H   + O2   -> HO2"),
        (rconst_symbols[11], {H: 1, HO2: 1}, {H: -1, HO2: -1, H2O2: 1}, "H   + HO2  -> H2O2"),
        (rconst_symbols[12], {H: 1, O2m: 1}, {H: -1, O2m: -1, HO2m: 1}, "H   + O2-  -> HO2-"),
        (rconst_symbols[13], {OH: 2}, {OH: -2, H2O2: 1}, "OH  + OH   -> H2O2"),
        (rconst_symbols[14], {OH: 1, H2: 1}, {OH: -1, H2: -1, H: 1, H2O: 1}, "OH  + H2   -> H    + H2O"),

        # # /* v16-v20*/
        # (0, 0, -1, -1, 0,    0, 1, 0, 1, 0,    0, 0, 0, 0),  # /* OH  + H2O2 -> H2O  + HO2 */
        # (0, 0, -1, 0, 1,    0, -1, 0, 1, 0,    0, 0, 0, 0),  # /* OH  + HO2  -> H2O  + O2  */
        # (0, 0, -1, 0, 1,   -1, 0, 0, 0, 1,    0, 0, 0, 0),   # /* OH  + O2-  -> OH-  + O2  */
        # (0, 0, 0, 1, 1,    0, -2, 0, 0, 0,    0, 0, 0, 0),   # /* 2HO2       -> H2O2 + O2  */
        # (0, 0, 0, 1, 1,   -1, -1, 0, 0, 1,    0, 0, 0, 0),   # /* HO2 + O2-  -> H2O2 + O2 + OH- */
        (rconst_symbols[15], {OH: 1, H2O2: 1}, {OH: -1, H2O2: -1, H2O: 1, HO2: 1}, "OH  + H2O2 -> H2O  + HO2"),
        (rconst_symbols[16], {OH: 1, HO2: 1}, {OH: -1, HO2: -1, H2O: 1, O2: 1}, "OH  + HO2  -> H2O  + O2"),
        (rconst_symbols[17], {OH: 1, O2m: 1}, {OH: -1, O2m: -1, OHm: 1, O2: 1}, "OH  + O2-  -> OH-  + O2"),
        (rconst_symbols[18], {HO2: 2}, {HO2: -2, H2O2: 1, O2: 1}, "2HO2       -> H2O2 + O2"),
        (
            rconst_symbols[19], {HO2: 1, O2m: 1}, {HO2: -1, O2m: -1, H2O2: 1, O2: 1, OHm: 1},
            "HO2 + O2-  -> H2O2 + O2 + OH-"),

        # # /* v21-v25*/
        # (0, 0, 0, 0, 0,    0, 0, 0, -1, 1,    0, 1, 0, 0),   # /* H2O         -> H+   + OH-  */
        # (0, 0, 0, 0, 0,    0, 0, 0, 1, -1,    0, -1, 0, 0),  # /* H+   + OH-  -> H2O         */
        # (0, 0, 0, -1, 0,    0, 0, 0, 0, 0,    1, 1, 0, 0),   # /* H2O2        -> H+   + HO2- */
        # (0, 0, 0, 1, 0,    0, 0, 0, 0, 0,   -1, -1, 0, 0),   # /* H+   + HO2- -> H2O2        */
        # (0, 0, 0, -1, 0,    0, 0, 0, 1, -1,    1, 0, 0, 0),  # /* H2O2 + OH-  -> HO2- + H2O  */
        (rconst_symbols[20], {H2O: 1}, {H2O: -1, Hp: 1, OHm: 1}, "H2O         -> H+   + OH-"),
        (rconst_symbols[21], {Hp: 1, OHm: 1}, {Hp: -1, OHm: -1, H2O: 1}, "H+   + OH-  -> H2O"),
        (rconst_symbols[22], {H2O2: 1}, {H2O2: -1, Hp: 1, HO2m: 1}, "H2O2        -> H+   + HO2-"),
        (rconst_symbols[23], {Hp: 1, HO2m: 1}, {Hp: -1, HO2m: -1, H2O2: 1}, "H+   + HO2- -> H2O2"),
        (rconst_symbols[24], {H2O2: 1, OHm: 1}, {H2O2: -1, OHm: -1, HO2m: 1, H2O: 1}, "H2O2 + OH-  -> HO2- + H2O"),

        # # /* v26-v30*/
        # (0, 0, 0, 1, 0,    0, 0, 0, -1, 1,   -1, 0, 0, 0),   # /* HO2- + H2O  -> H2O2 + OH-  */
        # (1, -1, 0, 0, 0,    0, 0, 0, 0, 0,    0, 1, 0, 0),   # /* H           -> e-   + H+   */
        # (-1, 1, 0, 0, 0,    0, 0, 0, 0, 0,    0, -1, 0, 0),  # /* e-   + H+   -> H           */
        # (-1, 1, 0, 0, 0,    0, 0, 0, -1, 1,    0, 0, 0, 0),  # /* e-   + H2O  -> H    + OH-  */
        # (1, -1, 0, 0, 0,    0, 0, 0, 1, -1,    0, 0, 0, 0),  # /* H    + OH-  -> e-   + H2O  */
        (rconst_symbols[25], {HO2m: 1, H2O: 1}, {HO2m: -1, H2O: -1, H2O2: 1, OHm: 1}, "HO2- + H2O  -> H2O2 + OH-"),
        (rconst_symbols[26], {H: 1}, {H: -1, em: 1, Hp: 1}, "H           -> e-   + H+"),
        (rconst_symbols[27], {em: 1, Hp: 1}, {em: -1, Hp: -1, H: 1}, "e-   + H+   -> H"),
        (rconst_symbols[28], {em: 1, H2O: 1}, {em: -1, H2O: -1, H: 1, OHm: 1}, "e-   + H2O  -> H    + OH-"),
        (rconst_symbols[29], {H: 1, OHm: 1}, {H: -1, OHm: -1, em: 1, H2O: 1}, "H    + OH-  -> e-   + H2O"),

        # # /* v31-v35*/
        # (0, 0, -1, 0, 0,    0, 0, 0, 0, 0,    0, 1, 1, 0),   # /* OH          -> H+   + O-   */
        # (0, 0, 1, 0, 0,    0, 0, 0, 0, 0,    0, -1, -1, 0),  # /* H+   + O-   -> OH          */
        # (0, 0, -1, 0, 0,    0, 0, 0, 1, -1,    0, 0, 1, 0),  # /* OH   + OH-  -> O-   + H2O  */
        # (0, 0, 1, 0, 0,    0, 0, 0, -1, 1,    0, 0, -1, 0),  # /* O-   + H2O  -> OH-  + OH   */
        # (0, 0, 0, 0, 0,    1, -1, 0, 0, 0,    0, 1, 0, 0),   # /* HO2         -> O2-  + H+   */
        (rconst_symbols[30], {OH: 1}, {OH: -1, Hp: 1, Om: 1}, "OH          -> H+   + O-"),
        (rconst_symbols[31], {Hp: 1, Om: 1}, {Hp: -1, Om: -1, OH: 1}, "H+   + O-   -> OH"),
        (rconst_symbols[32], {OH: 1, OHm: 1}, {OH: -1, OHm: -1, Om: 1, H2O: 1}, "OH   + OH-  -> O-   + H2O"),
        (rconst_symbols[33], {Om: 1, H2O: 1}, {Om: -1, H2O: -1, OHm: 1, OH: 1}, "O-   + H2O  -> OH-  + OH"),
        (rconst_symbols[34], {HO2: 1}, {HO2: -1, O2m: 1, Hp: 1}, "HO2         -> O2-  + H+"),

        # # /* v36-v40*/
        # (0, 0, 0, 0, 0,   -1, 1, 0, 0, 0,    0, -1, 0, 0),   # /* O2-  + H+   -> HO2         */
        # (0, 0, 0, 0, 0,    1, -1, 0, 1, -1,    0, 0, 0, 0),  # /* HO2  + OH-  -> O2-  + H2O  */
        # (0, 0, 0, 0, 0,   -1, 1, 0, -1, 1,    0, 0, 0, 0),   # /* O2-  + H2O  -> HO2  + OH-  */
        # (0, 1, 0, 0, 0,    0, 0, -1, 0, 1,    0, 0, -1, 0),  # /* O-   + H2   -> H    + OH-  */
        # (0, 0, 0, -1, 0,    1, 0, 0, 1, 0,    0, 0, -1, 0),  # /* O-   + H2O2 -> O2-  + H2O  */
        (rconst_symbols[35], {O2m: 1, Hp: 1}, {O2m: -1, Hp: -1, HO2: 1}, "O2-  + H+   -> HO2"),
        (rconst_symbols[36], {HO2: 1, OHm: 1}, {HO2: -1, OHm: -1, O2m: 1, H2O: 1}, "HO2  + OH-  -> O2-  + H2O"),
        (rconst_symbols[37], {O2m: 1, H2O: 1}, {O2m: -1, H2O: -1, HO2: 1, OHm: 1}, "O2-  + H2O  -> HO2  + OH-"),
        (rconst_symbols[38], {Om: 1, H2: 1}, {Om: -1, H2: -1, H: 1, OHm: 1}, "O-   + H2   -> H    + OH-"),
        (rconst_symbols[39], {Om: 1, H2O2: 1}, {Om: -1, H2O2: -1, O2m: 1, H2O: 1}, "O-   + H2O2 -> O2-  + H2O"),

        # # /* v41-v45*/
        # (0, 0, -1, 0, 0,    0, 1, 0, 0, 1,   -1, 0, 0, 0),   # /* OH   + HO2- -> OH-  + HO2  */
        # (0, 0, -1, 0, 0,    0, 0, 0, 0, 0,    1, 0, -1, 0),  # /* OH   + O-   -> HO2-        */
        # (-1, 0, 0, 0, 0,    0, 0, 0, 0, 1,   -1, 0, 1, 0),   # /* e-   + HO2- -> O-   + OH-  */
        # (-1, 0, 0, 0, 0,    0, 0, 0, 0, 2,    0, 0, -1, 0),  # /* e-   + O-   -> 2OH-        */
        # (0, 0, 0, 0, -1,    0, 0, 0, 0, 0,    0, 0, -1, 1),  # /* O-   + O2   -> O3-         */
        (rconst_symbols[40], {OH: 1, HO2m: 1}, {OH: -1, HO2m: -1, OHm: 1, HO2: 1}, "OH   + HO2- -> OH-  + HO2"),
        (rconst_symbols[41], {OH: 1, Om: 1}, {OH: -1, Om: -1, HO2m: 1}, "OH   + O-   -> HO2-"),
        (rconst_symbols[42], {em: 1, HO2m: 1}, {em: -1, HO2m: -1, Om: 1, OHm: 1}, "e-   + HO2- -> O-   + OH-"),
        (rconst_symbols[43], {em: 1, Om: 1}, {em: -1, Om: -1, OHm: 2}, "e-   + O-   -> 2OH-"),
        (rconst_symbols[44], {Om: 1, O2: 1}, {Om: -1, O2: -1, O3m: 1}, "O-   + O2   -> O3-"),

        # # /* v46-v50*/
        # (0, 0, 0, 0, 1,    0, 0, 0, 0, 0,    0, 0, 1, -1),   # /* O3-         -> O2   + O-   */
        # (0, 0, 0, 0, 0,    1, 0, 0, 0, 1,   -1, 0, -1, 0),   # /* O-   + HO2- -> O2-  + OH-  */
        # (0, 0, 0, 0, 1,   -1, 0, 0, 0, 2,    0, 0, -1, 0),   # /* O-   + O2-  -> 2OH- + O2   */
        # (0, 0, 1, -1, 1,    0, -1, 0, 1, 0,    0, 0, 0, 0),  # /* HO2  + H2O2 -> OH   + H2O + O2 */
        # (0, 0, 1, -1, 1,   -1, 0, 0, 0, 1,    0, 0, 0, 0),   # /* O2-  + H2O2 -> OH-  + OH  + O2 */
        (rconst_symbols[45], {O3m: 1}, {O3m: -1, O2: 1, Om: 1}, "O3-         -> O2   + O-"),
        (rconst_symbols[46], {Om: 1, HO2m: 1}, {Om: -1, HO2m: -1, O2m: 1, OHm: 1}, "O-   + HO2- -> O2-  + OH-"),
        (rconst_symbols[47], {Om: 1, O2m: 1}, {Om: -1, O2m: -1, OHm: 2, O2: 1}, "O-   + O2-  -> 2OH- + O2"),
        (rconst_symbols[48], {HO2: 1, H2O2: 1}, {HO2: -1, H2O2: -1, OH: 1, H2O: 1, O2: 1},
         "HO2  + H2O2 -> OH   + H2O + O2"),
        (rconst_symbols[49], {O2m: 1, H2O2: 1}, {O2m: -1, H2O2: -1, OHm: 1, OH: 1, O2: 1},
         "O2-  + H2O2 -> OH-  + OH  + O2"),

        # # /* irradiation: creation of aquaous electrons */
        # (1, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0)     # /*             -> e-              */
        (rconst_symbols[50], {}, {em: 1}, "? -> e-"),

    ]

    @classmethod
    @lru_cache
    def dCdt_exp(cls) -> list:
        """
        Calculate analytical expression for dCdt
        """

        # initialize dC/dt to zero
        dCdt = {C_symb: 0 for C_symb in cls.species_symbols}

        # loop over all reactions
        r_stoich: Dict
        for coeff, r_stoich, net_stoich, _ in cls.reactions:

            # calculate reaction rate, based on LHS of reaction equation
            # for ---(k)---> ? it will be k
            # for A ---(k)---> ? it will be k*A
            # for A + B ---(k)---> ? it will be k*A*B
            # for A + A ---(k)---> ? it will be k*A*A
            r = coeff * prod([rk ** p for rk, p in r_stoich.items()])

            # calculate ingredients of dC/dt including positive and negative terms from RHS and LHS
            for net_key, net_mult in net_stoich.items():
                dCdt[net_key] += net_mult * r

        return [dCdt[C_symb] for C_symb in cls.species_symbols]

    @classmethod
    @lru_cache
    def dCdt_f_lambda(cls) -> Callable[[Sequence[float], float, Sequence[float]], List[float]]:
        return sym.lambdify((cls.species_symbols, cls.t) + (cls.rconst_symbols,), cls.dCdt_exp())

    @classmethod
    def dCdt_f(cls, concentr: Sequence[float], t: float) -> List[float]:
        """
        Calculate function for dCdt
        """
        dCdt_lambda = cls.dCdt_f_lambda()
        return dCdt_lambda(concentr, t, cls.rconst_values)

    @classmethod
    @lru_cache
    def dCdt_Jac_f_lambda(cls) -> Callable[[Sequence[float], float, Sequence[float]], List[float]]:
        J = sym.Matrix(cls.dCdt_exp()).jacobian(cls.species_symbols)
        dCdt_jac_lambda = sym.lambdify((cls.species_symbols, cls.t) + (cls.rconst_symbols,), J)
        return dCdt_jac_lambda

    @classmethod
    def dCdt_Jac_f(cls, concentr: Sequence[float], t: float, model: Any, source: Any) -> List[float]:
        """
        Calculate function for Jacobian
        """
        dCdt_jac_lambda = cls.dCdt_Jac_f_lambda()
        return dCdt_jac_lambda(concentr, t, cls.rconst_values)
