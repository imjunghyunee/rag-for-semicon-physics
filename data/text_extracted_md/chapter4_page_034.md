At the same time, because of this redistribution, the net electron concentration in the conduction band is not simply equal to the donor concentration plus the intrinsic electron concentration.

## Objective: Calculate the thermal-equilibrium electron and hole concentrations in germanium for a given doping concentration

Consider a germanium sample at \( T = 300 \, \text{K} \) in which \( N_D = 2 \times 10^{14} \, \text{cm}^{-3} \) and \( N_A = 0 \). Assume that \( n_i = 2.4 \times 10^{13} \, \text{cm}^{-3} \).

### Solution

Again, from Equation (4.60), the majority carrier electron concentration is

\[
n_0 = \frac{2 \times 10^{14}}{2} + \sqrt{\left(\frac{2 \times 10^{14}}{2}\right)^2 + (2.4 \times 10^{13})^2} \approx 2.028 \times 10^{14} \, \text{cm}^{-3}
\]

The minority carrier hole concentration is

\[
p_0 = \frac{n_i^2}{n_0} = \frac{(2.4 \times 10^{13})^2}{2.028 \times 10^{14}} = 2.84 \times 10^{12} \, \text{cm}^{-3}
\]

### Comment

If the donor impurity concentration is not too different in magnitude from the intrinsic carrier concentration, then the thermal-equilibrium majority carrier electron concentration is influenced by the intrinsic concentration.

### EXERCISE PROBLEM

**Ex 4.10** Repeat Example 4.10 for (a) \( T = 250 \, \text{K} \) and (b) \( T = 350 \, \text{K} \). (c) What can be said about a very low-doped material as the temperature increases?

----

We have seen that the intrinsic carrier concentration \( n_i \) is a very strong function of temperature. As the temperature increases, additional electron-hole pairs are thermally generated so that the \( n_i^2 \) term in Equation (4.60) may begin to dominate. The semiconductor will eventually lose its extrinsic characteristics. Figure 4.16 shows the electron concentration versus temperature in silicon doped with \( 5 \times 10^{14} \) donors per cm\(^3\). As the temperature increases, we can see where the intrinsic concentration begins to dominate. Also shown is the partial ionization, or the onset of freeze-out, at the low temperature.

### Thermal-Equilibrium Hole Concentration

If we reconsider Equation (4.58) and express \( n_i^2/p_0 \), then we have

\[
\frac{n_i^2}{p_0} + N_D = p_0 + N_A \tag{4.61a}
\]

which we can write as

\[
p_0^2 - (N_A - N_D)p_0 - n_i^2 = 0 \tag{4.61b}
\]