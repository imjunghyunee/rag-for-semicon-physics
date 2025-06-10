### 4.1 Charge Carriers in Semiconductors

or

\[
f(E) = \exp\left[\frac{-0.25 + (0.0259/2)}{0.0259}\right] = 3.90 \times 10^{-5}
\]

The electron concentration is given by

\[
n_0 = N_c \exp\left[\frac{-(E_c - E_F)}{kT}\right] = (2.8 \times 10^{19}) \exp\left[\frac{-0.25}{0.0259}\right]
\]

or

\[
n_0 = 1.80 \times 10^{15} \text{ cm}^{-3}
\]

----

**Comment**

The probability of a state being occupied can be quite small, but the fact that there are a large number of states means that the electron concentration is a reasonable value.

----

**EXERCISE PROBLEM**

**Ex 4.1** Determine the probability that a quantum state at energy \( E = E_c + kT \) is occupied by an electron, and calculate the electron concentration in GaAs at \( T = 300 \, K \) if the Fermi energy level is 0.25 eV below \( E_c \).

----

### Thermal-Equilibrium Hole Concentration

The thermal-equilibrium concentration of holes in the valence band is found by integrating Equation (4.2) over the valence-band energy, or

\[
p_0 = \int g_v(E)[1 - f(E)] \, dE \tag{4.12}
\]

We may note that

\[
1 - f(E) = \frac{1}{1 + \exp\left[\frac{E - E_F}{kT}\right]} \tag{4.13a}
\]

For energy states in the valence band, \( E \ll E_F \). If \( (E_F - E) \gg kT \) (the Fermi function is still assumed to be within the bandgap), then we have a slightly different form of the Boltzmann approximation. Equation (4.13a) may be written as

\[
1 - f(E) = \frac{1}{1 + \exp\left[\frac{E_F - E}{kT}\right]} \approx \exp\left[\frac{-(E_F - E)}{kT}\right] \tag{4.13b}
\]

Applying the Boltzmann approximation of Equation (4.13b) to Equation (4.12), we find the thermal-equilibrium concentration of holes in the valence band is

\[
p_0 = \int \frac{4\pi (2m_p)^{3/2}}{h^3} \sqrt{E_v - E} \exp\left[\frac{-(E_F - E)}{kT}\right] \, dE \tag{4.14}
\]

where the lower limit of integration is taken as minus infinity instead of the bottom of the valence band. The exponential term decays fast enough so that this approximation is valid.