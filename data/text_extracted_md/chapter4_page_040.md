```markdown
## Objective: Determine the Fermi energy level and the maximum doping concentration at which the Boltzmann approximation is still valid.

Consider p-type silicon, at \( T = 300 \, \text{K} \), doped with boron. We may assume that the limit of the Boltzmann approximation occurs when \( E_F - E_i = 3kT \). (See Section 4.1.2.)

### Solution

From Table 4.3, we find the ionization energy is \( E_c - E_i = 0.045 \, \text{eV} \) for boron in silicon. If we assume that \( E_F \approx E_{\text{acceptor}} \), then from Equation (4.68), the position of the Fermi level at the maximum doping is given by

\[
E_F - E_i = \frac{E_c - E_i}{2} - (E_c - E_i) = kT \ln \left( \frac{N_a}{n_i} \right)
\]

or

\[
0.56 - 0.045 - 3(0.0259) = 0.437 = (0.0259) \ln \left( \frac{N_a}{n_i} \right)
\]

We can then solve for the doping as

\[
N_a = n_i \exp \left( \frac{0.437}{0.0259} \right) = 3.2 \times 10^{17} \, \text{cm}^{-3}
\]

### Comment

If the acceptor (or donor) concentration in silicon is greater than approximately \( 3 \times 10^{17} \, \text{cm}^{-3} \), then the Boltzmann approximation of the distribution function becomes less valid and the equations for the Fermi-level position are no longer quite as accurate.

### EXERCISE PROBLEM

**Ex 4.13** Consider n-type silicon at \( T = 300 \, \text{K} \) doped with arsenic. Determine the maximum doping at which the Boltzmann approximation is still valid. Assume the limit is such that \( E_F - E_i = 3kT. \left( c_{ud} = 0.01 \times 0.70 \, \text{e} = 0.1 \, \text{suV} \right) \)

The intrinsic carrier concentration, \( n_i \), in Equations (4.65) and (4.68), is a strong function of temperature, so that \( E_F \) is a function of temperature also. Figure 4.19 shows the variation of the Fermi energy level in silicon with temperature for several donor and acceptor concentrations. As the temperature increases, \( n_i \) increases, and \( E_F \) moves closer to the intrinsic Fermi level. At high temperature, the semiconductor material begins to lose its extrinsic characteristics and begins to behave more like an intrinsic semiconductor. At the very low temperature, freeze-out occurs; the Boltzmann approximation is no longer valid and the equations we derived for the Fermi-level position no longer apply. At the low temperature where freeze-out occurs, the Fermi level goes above \( E_F \) for the n-type material and below \( E_F \) for the p-type material. At absolute zero degrees, all energy states below \( E_F \) are full and all energy states above \( E_F \) are empty.

### 4.6.3 Relevance of the Fermi Energy

We have been calculating the position of the Fermi energy level as a function of doping concentrations and temperature. This analysis may seem somewhat arbitrary and fictitious. However, these relations do become significant later in our discussion of pn junctions and the other semiconductor devices we consider. An important...
```