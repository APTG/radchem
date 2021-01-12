class RadChemModel():

    nspecies = 14  # number of species tracked
    neq = 51   # number of equations

    symbol = {
        "e-": 0,
        "H": 1,
        "OH": 2,
        "H2O2": 3,
        "O2": 4,

        "O2-": 5,
        "HO2": 6,
        "H2": 7,
        "H2O": 8,
        "OH-": 9,

        "HO2-": 10,
        "H+": 11,
        "O-": 12,
        "O3-": 13
    }

    # /* G-values at 25 deg C [#/100eV] */
    gval = (  # gval[NSPECIES] = (
        2.645,   # /* A0  : e-   */
        0.572,   # /* A1  : H    */
        2.819,   # /* A2  : OH   */
        0.646,   # /* A3  : H2O2 */
        0,       # /* A4  : O2   */

        0,       # /* A5  : O2-  */
        0,       # /* A6  : HO2  */
        0.447,   # /* A7  : H2   */
        -4.541,  # /* A8  : H2O  */
        0.430,   # /* A9  : OH-  */

        0,       # /* A10 : HO2- */
        3.075,   # /* A11 : H+   */
        0.430,   # /* A12 : O-   */
        0        # /* A13 : O3-  */
    )

    nmatrix = (   # nmatrix[NEQ][NSPECIES]
        # /* v1-v5*/
        # /*0           4     5           9    10        14*/
        (-2, 0, 0, 0, 0,    0, 0, 1, 0, 2,    0, 0, 0, 0),   # /* e-  + e-   -> H2  + 2OH- */
        (-1, -1, 0, 0, 0,    0, 0, 1, 0, 1,    0, 0, 0, 0),  # /* e-  + H    -> H2  + OH-  */
        (-1, 0, -1, 0, 0,    0, 0, 0, 0, 1,    0, 0, 0, 0),  # /* e-  + OH   -> OH-        */
        (-1, 0, 1, -1, 0,    0, 0, 0, 0, 1,    0, 0, 0, 0),  # /* e-  + H2O2 -> OH- + OH   */
        (-1, 0, 0, 0, -1,    1, 0, 0, 0, 0,    0, 0, 0, 0),  # /* e-  + O2   -> O2-        */

        # /* v5-v10*/
        (-1, 0, 0, 0, 0,   -1, 0, 0, 0, 1,    1, 0, 0, 0),   # /* e-  + O2-  -> HO2- + OH- */
        (-1, 0, 0, 0, 0,    0, -1, 0, 0, 0,    1, 0, 0, 0),  # /* e-  + HO2  -> HO2-       */
        (0, -2, 0, 0, 0,    0, 0, 1, 0, 0,    0, 0, 0, 0),   # /* 2H         -> H2         */
        (0, -1, -1, 0, 0,    0, 0, 0, 1, 0,    0, 0, 0, 0),  # /* H   + OH   -> H2O        */
        (0, -1, 1, -1, 0,    0, 0, 0, 1, 0,    0, 0, 0, 0),  # /* H   + H2O2 -> OH   + H2O */

        # /* v11-v15*/
        (0, -1, 0, 0, -1,    0, 1, 0, 0, 0,    0, 0, 0, 0),  # /* H   + O2   -> HO2        */
        (0, -1, 0, 1, 0,    0, -1, 0, 0, 0,    0, 0, 0, 0),  # /* H   + HO2  -> H2O2       */
        (0, -1, 0, 0, 0,   -1, 0, 0, 0, 0,    1, 0, 0, 0),   # /* H   + O2-  -> HO2-       */
        (0, 0, -2, 1, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0),   # /* OH  + OH   -> H2O2       */
        (0, 1, -1, 0, 0,    0, 0, -1, 1, 0,    0, 0, 0, 0),  # /* OH  + H2   -> H    + H2O */

        # /* v16-v20*/
        (0, 0, -1, -1, 0,    0, 1, 0, 1, 0,    0, 0, 0, 0),  # /* OH  + H2O2 -> H2O  + HO2 */
        (0, 0, -1, 0, 1,    0, -1, 0, 1, 0,    0, 0, 0, 0),  # /* OH  + HO2  -> H2O  + O2  */
        (0, 0, -1, 0, 1,   -1, 0, 0, 0, 1,    0, 0, 0, 0),   # /* OH  + O2-  -> OH-  + O2  */
        (0, 0, 0, 1, 1,    0, -2, 0, 0, 0,    0, 0, 0, 0),   # /* 2HO2       -> H2O2 + O2  */
        (0, 0, 0, 1, 1,   -1, -1, 0, 0, 1,    0, 0, 0, 0),   # /* HO2 + O2-  -> H2O2 + O2 + OH- */

        # /* v21-v25*/
        (0, 0, 0, 0, 0,    0, 0, 0, -1, 1,    0, 1, 0, 0),   # /* H2O         -> H+   + OH-  */
        (0, 0, 0, 0, 0,    0, 0, 0, 1, -1,    0, -1, 0, 0),  # /* H+   + OH-  -> H2O         */
        (0, 0, 0, -1, 0,    0, 0, 0, 0, 0,    1, 1, 0, 0),   # /* H2O2        -> H+   + HO2- */
        (0, 0, 0, 1, 0,    0, 0, 0, 0, 0,   -1, -1, 0, 0),   # /* H+   + HO2- -> H2O2        */
        (0, 0, 0, -1, 0,    0, 0, 0, 1, -1,    1, 0, 0, 0),  # /* H2O2 + OH-  -> HO2- + H2O  */

        # /* v26-v30*/
        (0, 0, 0, 1, 0,    0, 0, 0, -1, 1,   -1, 0, 0, 0),   # /* HO2- + H2O  -> H2O2 + OH-  */
        (1, -1, 0, 0, 0,    0, 0, 0, 0, 0,    0, 1, 0, 0),   # /* H           -> e-   + H+   */
        (-1, 1, 0, 0, 0,    0, 0, 0, 0, 0,    0, -1, 0, 0),  # /* e-   + H+   -> H           */
        (-1, 1, 0, 0, 0,    0, 0, 0, -1, 1,    0, 0, 0, 0),  # /* e-   + H2O  -> H    + OH-  */
        (1, -1, 0, 0, 0,    0, 0, 0, 1, -1,    0, 0, 0, 0),  # /* H    + OH-  -> e-   + H2O  */

        # /* v31-v35*/
        (0, 0, -1, 0, 0,    0, 0, 0, 0, 0,    0, 1, 1, 0),   # /* OH          -> H+   + O-   */
        (0, 0, 1, 0, 0,    0, 0, 0, 0, 0,    0, -1, -1, 0),  # /* H+   + O-   -> OH          */
        (0, 0, -1, 0, 0,    0, 0, 0, 1, -1,    0, 0, 1, 0),  # /* OH   + OH-  -> O-   + H2O  */
        (0, 0, 1, 0, 0,    0, 0, 0, -1, 1,    0, 0, -1, 0),  # /* O-   + H2O  -> OH-  + OH   */
        (0, 0, 0, 0, 0,    1, -1, 0, 0, 0,    0, 1, 0, 0),   # /* HO2         -> O2-  + H+   */


        # /* v36-v40*/
        (0, 0, 0, 0, 0,   -1, 1, 0, 0, 0,    0, -1, 0, 0),   # /* O2-  + H+   -> HO2         */
        (0, 0, 0, 0, 0,    1, -1, 0, 1, -1,    0, 0, 0, 0),  # /* HO2  + OH-  -> O2-  + H2O  */
        (0, 0, 0, 0, 0,   -1, 1, 0, -1, 1,    0, 0, 0, 0),   # /* O2-  + H2O  -> HO2  + OH-  */
        (0, 1, 0, 0, 0,    0, 0, -1, 0, 1,    0, 0, -1, 0),  # /* O-   + H2   -> H    + OH-  */
        (0, 0, 0, -1, 0,    1, 0, 0, 1, 0,    0, 0, -1, 0),  # /* O-   + H2O2 -> O2-  + H2O  */


        # /* v41-v45*/
        (0, 0, -1, 0, 0,    0, 1, 0, 0, 1,   -1, 0, 0, 0),   # /* OH   + HO2- -> OH-  + HO2  */
        (0, 0, -1, 0, 0,    0, 0, 0, 0, 0,    1, 0, -1, 0),  # /* OH   + O-   -> HO2-        */
        (-1, 0, 0, 0, 0,    0, 0, 0, 0, 1,   -1, 0, 1, 0),   # /* e-   + HO2- -> O-   + OH-  */
        (-1, 0, 0, 0, 0,    0, 0, 0, 0, 2,    0, 0, -1, 0),  # /* e-   + O-   -> 2OH-        */
        (0, 0, 0, 0, -1,    0, 0, 0, 0, 0,    0, 0, -1, 1),  # /* O-   + O2   -> O3-         */

        # /* v46-v50*/
        (0, 0, 0, 0, 1,    0, 0, 0, 0, 0,    0, 0, 1, -1),   # /* O3-         -> O2   + O-   */
        (0, 0, 0, 0, 0,    1, 0, 0, 0, 1,   -1, 0, -1, 0),   # /* O-   + HO2- -> O2-  + OH-  */
        (0, 0, 0, 0, 1,   -1, 0, 0, 0, 2,    0, 0, -1, 0),   # /* O-   + O2-  -> 2OH- + O2   */
        (0, 0, 1, -1, 1,    0, -1, 0, 1, 0,    0, 0, 0, 0),  # /* HO2  + H2O2 -> OH   + H2O + O2 */
        (0, 0, 1, -1, 1,   -1, 0, 0, 0, 1,    0, 0, 0, 0),   # /* O2-  + H2O2 -> OH-  + OH  + O2 */

        # /* irradiation: creation of aquaous electrons */
        (1, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0)     # /*             -> e-              */
    )

    rconst = (
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