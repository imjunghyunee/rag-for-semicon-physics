## EXAMPLE 8.6

**Objective:** Determine the relative magnitudes of the ideal reverse-saturation current density and the generation current density in a reverse-biased pn junction.

Consider a silicon pn junction at \( T = 300 \, \text{K} \) with parameters \( D_n = 25 \, \text{cm}^2/\text{s}, D_p = 10 \, \text{cm}^2/\text{s}, N_d = N_a = 10^{16} \, \text{cm}^{-3} \), and \( \tau_n = \tau_p = \tau_0 = 5 \times 10^{-7} \, \text{s} \). Assume the diode is reverse biased at \( V_R = 5 \, \text{V} \).

### Solution

The ideal reverse-saturation current density was calculated in Example 8.2 and was found to be \( J_0 = 4.155 \times 10^{-11} \, \text{A/cm}^2 \).

The built-in potential is found as

\[
V_{bi} = V_T \ln \left( \frac{N_a N_d}{n_i^2} \right) = (0.0259) \ln \left( \frac{(10^{16})(10^{16})}{(1.5 \times 10^{10})^2} \right) = 0.695 \, \text{V}
\]

The depletion width is found to be

\[
W = \left[ \frac{2 \varepsilon_s (V_{bi} + V_R)}{e} \left( \frac{N_a + N_d}{N_a N_d} \right) \right]^{1/2}
\]

\[
= \left[ \frac{(2)(11.7)(8.85 \times 10^{-14})(0.695 + 5)}{1.6 \times 10^{-19}} \right] \left[ \frac{10^{16} + 10^{16}}{(10^{16})(10^{16})} \right]^{1/2}
\]

\[
= 1.214 \times 10^{-4} \, \text{cm}
\]

The generation current density is then found to be

\[
J_{gen} = \frac{e n_i W}{2 \tau_0} = \frac{(1.6 \times 10^{-19})(1.5 \times 10^{10})(1.214 \times 10^{-4})}{2(5 \times 10^{-7})}
\]

or

\[
J_{gen} = 2.914 \times 10^{-7} \, \text{A/cm}^2
\]

The ratio of the two currents is

\[
\frac{J_{gen}}{J_0} = \frac{2.914 \times 10^{-7}}{4.155 \times 10^{-11}} = 7 \times 10^3
\]

### Comment

Comparing the solutions for the two current densities, it is obvious that, for the silicon pn junction diode at room temperature, the generation current density is approximately four orders of magnitude larger than the ideal saturation current density. The generation current is the dominant reverse-biased current in a silicon pn junction diode.

### EXERCISE PROBLEM

**Ex 8.6** Consider a GaAs pn junction diode at \( T = 300 \, \text{K} \) with parameters \( N_d = 8 \times 10^{16} \, \text{cm}^{-3}, N_a = 2 \times 10^{16} \, \text{cm}^{-3}, D_n = 207 \, \text{cm}^2/\text{s}, D_p = 9.80 \, \text{cm}^2/\text{s}, \) and \( \tau_n = \tau_p = \tau_0 = 5 \times 10^{-8} \, \text{s} \). (a) Calculate the ideal reverse-biased saturation current density. (b) Find the reverse-biased generation current density if the diode is reverse biased at \( V_R = 5 \, \text{V} \). (c) Determine the ratio of \( J_{gen} \) to \( J_0 \).