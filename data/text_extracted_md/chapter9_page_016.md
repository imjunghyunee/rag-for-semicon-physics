# CHAPTER 9 Metal–Semiconductor and Semiconductor Heterojunctions

The reverse-saturation current density of the Schottky barrier diode was given by Equation (9.26) and is

\[
J_{sT} = A^* T^2 \exp \left( \frac{-e \phi_{Bn}}{kT} \right)
\]

The ideal reverse-saturation current density of the pn junction diode can be written as

\[
J_s = \frac{e D_n n_{po}}{L_n} + \frac{e D_p p_{no}}{L_p}
\]

(9.28)

The form of the two equations is vastly different, and the current mechanism in the two devices is different. The current in a pn junction is determined by the diffusion of minority carriers while the current in a Schottky barrier diode is determined by thermionic emission of majority carriers over a potential barrier.

## EXAMPLE 9.5

**Objective:** Calculate the ideal reverse-saturation current densities of a Schottky barrier diode and a pn junction diode.

Consider a tungsten barrier on silicon with a measured barrier height of \(\phi_{Bn} = 0.67 \, \text{eV}\). The effective Richardson constant is \(A^* = 114 \, \text{A/K}^2\cdot\text{cm}^2\). Let \(T = 300 \, \text{K}\).

### Solution

If we neglect the barrier lowering effect, we have for the Schottky barrier diode

\[
J_{sT} = A^* T^2 \exp \left( \frac{-e \phi_{Bn}}{kT} \right) = (114)(300)^2 \exp \left( \frac{-0.67}{0.0259} \right) = 5.98 \times 10^{-5} \, \text{A/cm}^2
\]

Consider a silicon pn junction with the following parameters at \(T = 300 \, \text{K}\).

\[
\begin{align*}
N_a &= 10^{18} \, \text{cm}^{-3} \\
N_d &= 10^{16} \, \text{cm}^{-3} \\
D_p &= 10 \, \text{cm}^2/\text{s} \\
D_n &= 25 \, \text{cm}^2/\text{s} \\
\tau_{po} &= 10^{-7} \, \text{s} \\
\tau_{no} &= 10^{-7} \, \text{s}
\end{align*}
\]

We can then calculate the following parameters:

\[
\begin{align*}
L_p &= 1.0 \times 10^{-3} \, \text{cm} \\
L_n &= 1.58 \times 10^{-3} \, \text{cm} \\
p_{no} &= 2.25 \times 10^2 \, \text{cm}^{-3} \\
n_{po} &= 2.25 \times 10^4 \, \text{cm}^{-3}
\end{align*}
\]

The ideal reverse-saturation current density of the pn junction diode can be determined from Equation (9.28) as

\[
\begin{align*}
J_s &= \left( \frac{1.6 \times 10^{-19}(25)(2.25 \times 10^4)}{1.58 \times 10^{-3}} \right) + \left( \frac{1.6 \times 10^{-19}(10)(2.25 \times 10^2)}{1.0 \times 10^{-3}} \right) \\
&= 5.7 \times 10^{-13} + 3.6 \times 10^{-11} = 3.66 \times 10^{-11} \, \text{A/cm}^2
\end{align*}
\]

### Comment

The ideal reverse-saturation current density of the Schottky barrier junction is orders of magnitude larger than that of the ideal pn junction diode.

### EXERCISE PROBLEM

**Ex 9.5** Using the results of Example 9.5, determine the forward-bias voltages required to produce a current of 10 μA in each diode. Assume each cross-sectional area is \(10^{-4} \, \text{cm}^2\).