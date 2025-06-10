## 5.3 Graded Impurity Distribution

Positive and negative charge induces an electric field that is in a direction to oppose the diffusion process. When equilibrium is reached, the mobile carrier concentration is not exactly equal to the fixed impurity concentration and the induced electric field prevents any further separation of charge. In most cases of interest, the space charge induced by this diffusion process is a small fraction of the impurity concentration, thus the mobile carrier concentration is not too different from the impurity dopant density.

The electric potential \( \phi \) is related to electron potential energy by the charge \((-e)\), so we can write

\[
\phi = + \frac{1}{e} (E_F - E_{Fn})
\]

(5.37)

The electric field for the one-dimensional situation is defined as

\[
E_x = -\frac{d\phi}{dx} = \frac{1}{e} \frac{dE_{Fn}}{dx}
\]

(5.38)

If the intrinsic Fermi-level changes as a function of distance through a semiconductor in thermal equilibrium, an electric field exists in the semiconductor.

If we assume a quasi-neutrality condition in which the electron concentration is almost equal to the donor impurity concentration, then we can still write

\[
n_0 = n_i \exp \left[ \frac{E_F - E_{Fi}}{kT} \right] \approx N_d(x)
\]

(5.39)

Solving for \(E_F - E_{Fi}\), we obtain

\[
E_F - E_{Fi} = kT \ln \left( \frac{N_d(x)}{n_i} \right)
\]

(5.40)

The Fermi level is constant for thermal equilibrium so when we take the derivative with respect to \(x\) we obtain

\[
-\frac{dE_{Fi}}{dx} = -\frac{kT}{N_d(x)} \frac{dN_d(x)}{dx}
\]

(5.41)

The electric field can then be written, combining Equations (5.41) and (5.38), as

\[
E_x = -\left( \frac{kT}{e} \right) \frac{1}{N_d(x)} \frac{dN_d(x)}{dx}
\]

(5.42)

Since we have an electric field, there will be a potential difference through the semiconductor due to the nonuniform doping.

----

**Objective:** Determine the induced electric field in a semiconductor in thermal equilibrium, given a linear variation in doping concentration.

**Example 5.6**

Assume that the donor concentration in an n-type semiconductor at \(T = 300 \, \text{K}\) is given by

\[
N_d(x) = 10^{16} - 10^{19} \times x \quad (\text{cm}^{-3})
\]

where \(x\) is given in cm and ranges between \(0 \leq x \leq 1 \, \mu\text{m}\).