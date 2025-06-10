# CHAPTER 4 The Semiconductor in Equilibrium

!Graph

**Figure 4.16** Electron concentration versus temperature showing the three regions: partial ionization, extrinsic, and intrinsic.

Using the quadratic formula, the hole concentration is given by

\[
p_0 = \frac{N_A}{2} + \sqrt{\left(\frac{N_A - N_D}{2}\right)^2 + n_i^2}
\]

(4.62)

where the positive sign, again, must be used. Equation (4.62) is used to calculate the thermal-equilibrium majority carrier hole concentration in a p-type semiconductor, or when \(N_A > N_D\). This equation also applies for \(N_D = 0\).

## EXAMPLE 4.11

**Objective:** Calculate the thermal-equilibrium electron and hole concentrations in a compensated p-type semiconductor.

Consider a silicon semiconductor at \(T = 300 \, K\) in which \(N_D = 10^{16} \, \text{cm}^{-3}\) and \(N_A = 3 \times 10^{15} \, \text{cm}^{-3}\). Assume \(n_i = 1.5 \times 10^{10} \, \text{cm}^{-3}\).

### Solution

Since \(N_D > N_A\), the compensated semiconductor is p-type and the thermal-equilibrium majority carrier hole concentration is given by Equation (4.62) as

\[
p_0 = \frac{10^{16} - 3 \times 10^{15}}{2} + \sqrt{\left(\frac{10^{16} - 3 \times 10^{15}}{2}\right)^2 + (1.5 \times 10^{10})^2}
\]

so that

\[
p_0 \approx 7 \times 10^{15} \, \text{cm}^{-3}
\]

The minority carrier electron concentration is

\[
n_0 = \frac{n_i^2}{p_0} = \frac{(1.5 \times 10^{10})^2}{7 \times 10^{15}} = 3.21 \times 10^4 \, \text{cm}^{-3}
\]

### Comment

If we assume complete ionization and if \((N_D - N_A) \gg n_i\), then the majority carrier hole concentration is, to a very good approximation, just the difference between the acceptor and donor concentrations.