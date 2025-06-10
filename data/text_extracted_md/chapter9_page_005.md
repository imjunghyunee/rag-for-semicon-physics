# 9.1 The Schottky Barrier Diode

The electric field can then be written as

\[
E = -\frac{eN_d}{\varepsilon_s} (x_n - x)
\]

(9.6)

which is a linear function of distance, for the uniformly doped semiconductor, and reaches a peak value at the metal–semiconductor interface. Since the E-field is zero inside the metal, a negative surface charge must exist in the metal at the metal–semiconductor junction.

The space charge region width, \( W \), may be calculated as we do for the pn junction. The result is identical to that of a one-sided p\(^+\)n junction. For the uniformly doped semiconductor, we have

\[
W = x_n = \left[ \frac{2\varepsilon_s(V_{bi} + V_R)}{eN_d} \right]^{1/2}
\]

(9.7)

where \( V_R \) is the magnitude of the applied reverse-biased voltage. We are again assuming an abrupt junction approximation.

----

**Objective:** Determine the theoretical barrier height, built-in potential barrier, and maximum electric field in a metal–semiconductor diode for zero applied bias.

**Example 9.1**

Consider a contact between tungsten and n-type silicon doped to \( N_d = 10^{16} \, \text{cm}^{-3} \) at \( T = 300 \, \text{K} \).

## Solution

The metal work function for tungsten (W) from Table 9.1 is \( \phi_m = 4.55 \, \text{V} \) and the electron affinity for silicon from Table 9.2 is \( \chi = 4.01 \, \text{V} \). The barrier height is then

\[
\phi_{bo} = \phi_m - \chi = 4.55 - 4.01 = 0.54 \, \text{V}
\]

where \( \phi_{bo} \) is the ideal Schottky barrier height. We can calculate \( \phi_b \) as

\[
\phi_b = \frac{kT}{e} \ln \left( \frac{N_c}{N_d} \right) = 0.0259 \ln \left( \frac{2.8 \times 10^{19}}{10^{16}} \right) = 0.206 \, \text{V}
\]

Then

\[
V_{bi} = \phi_{bo} - \phi_b = 0.54 - 0.206 = 0.334 \, \text{V}
\]

The space charge width at zero bias is

\[
x_n = \left[ \frac{2\varepsilon_s V_{bi}}{eN_d} \right]^{1/2} = \left[ \frac{2(11.7 \times 8.85 \times 10^{-14})(0.334)}{(1.6 \times 10^{-19})(10^{16})} \right]^{1/2}
\]

or

\[
x_n = 0.208 \times 10^{-4} \, \text{cm}
\]

Then the maximum electric field is

\[
|E_{\text{max}}| = \frac{eN_d x_n}{\varepsilon_s} = \frac{(1.6 \times 10^{-19})(10^{16})(0.208 \times 10^{-4})}{(11.7)(8.85 \times 10^{-14})}
\]

or finally

\[
|E_{\text{max}}| = 3.21 \times 10^{4} \, \text{V/cm}
\]