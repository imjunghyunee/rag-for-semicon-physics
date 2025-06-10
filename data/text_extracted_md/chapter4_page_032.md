# 4.5 Charge Neutrality

## Thermal-Equilibrium Electron Concentration

If we assume complete ionization, \( n_d \) and \( p_a \) are both zero, and Equation (4.57) becomes:

\[
n_0 + N_a = p_0 + \frac{n_i^2}{n_0} = N_d
\]

(4.58)

If we express \( p_0 \) as \( n_i^2/n_0 \), then Equation (4.58) can be written as:

\[
n_0 + N_a = \frac{n_i^2}{n_0} + N_d
\]

(4.59a)

which in turn can be written as:

\[
n_0^2 = (N_d - N_a)n_0 - n_i^2 = 0
\]

(4.59b)

The electron concentration \( n_0 \) can be determined using the quadratic formula, or:

\[
n_0 = \frac{(N_d - N_a)}{2} + \sqrt{\left(\frac{N_d - N_a}{2}\right)^2 + n_i^2}
\]

(4.60)

The **positive sign** in the quadratic formula must be used, since, in the limit of an intrinsic semiconductor when \( N_d = N_a = 0 \), the electron concentration must be a positive quantity, or \( n_0 = n_i \).

Equation (4.60) is used to calculate the electron concentration in an n-type semiconductor, or when \( N_d > N_a \). Although Equation (4.60) was derived for a compensated semiconductor, the equation is also valid for \( N_a = 0 \).

----

## Objective: Determine the thermal-equilibrium electron and hole concentrations in silicon at \( T = 300 \, K \) for given doping concentrations.

(a) Let \( N_d = 10^{16} \, \text{cm}^{-3} \) and \( N_a = 0 \).

(b) Let \( N_d = 5 \times 10^{15} \, \text{cm}^{-3} \) and \( N_a = 2 \times 10^{15} \, \text{cm}^{-3} \).

Recall that \( n_i = 1.5 \times 10^{10} \, \text{cm}^{-3} \) in silicon at \( T = 300 \, K \).

### Solution

(a) From Equation (4.60), the majority carrier electron concentration is:

\[
n_0 = \frac{10^{16}}{2} + \sqrt{\left(\frac{10^{16}}{2}\right)^2 + (1.5 \times 10^{10})^2} \approx 10^{16} \, \text{cm}^{-3}
\]

The minority carrier hole concentration is found to be:

\[
p_0 = \frac{n_i^2}{n_0} = \frac{(1.5 \times 10^{10})^2}{10^{16}} = 2.25 \times 10^4 \, \text{cm}^{-3}
\]

(b) Again, from Equation (4.60), the majority carrier electron concentration is:

\[
n_0 = \frac{5 \times 10^{15} - 2 \times 10^{15}}{2} + \sqrt{\left(\frac{5 \times 10^{15} - 2 \times 10^{15}}{2}\right)^2 + (1.5 \times 10^{10})^2} \approx 3 \times 10^{15} \, \text{cm}^{-3}
\]

The minority carrier hole concentration is:

\[
p_0 = \frac{n_i^2}{n_0} = \frac{(1.5 \times 10^{10})^2}{3 \times 10^{15}} = 7.5 \times 10^4 \, \text{cm}^{-3}
\]