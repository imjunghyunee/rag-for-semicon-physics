# CHAPTER 8 The pn Junction Diode

!Figure 8.8

**Figure 8.8** Ideal \( I \text{-} V \) characteristic of a pn junction diode.

so that Equation (8.25) may be written as

\[
J = J_s \left[ \exp \left( \frac{eV_a}{kT} \right) - 1 \right]
\]

(8.27)

Equation (8.27), known as the ideal-diode equation, gives a good description of the current–voltage characteristics of the pn junction over a wide range of currents and voltages. Although Equation (8.27) was derived assuming a forward-bias voltage (\( V_a > 0 \)), there is nothing to prevent \( V_a \) from being negative (reverse bias). Equation (8.27) is plotted in Figure 8.8 as a function of forward-bias voltage \( V_a \). If the voltage \( V_a \) becomes negative (reverse bias) by a few kT/eV, then the reverse-biased current density becomes independent of the reverse-biased voltage. The parameter \( J_s \) is then referred to as the reverse-saturation current density. The current–voltage characteristics of the pn junction diode are obviously not bilateral.

----

**EXAMPLE 8.2**

**Objective:** Determine the ideal reverse-saturation current density in a silicon pn junction at \( T = 300 \, \text{K} \).

Consider the following parameters in a silicon pn junction:

\[
\begin{align*}
N_d = N_a &= 10^{16} \, \text{cm}^{-3} \\
D_n &= 25 \, \text{cm}^2/\text{s} \\
D_p &= 10 \, \text{cm}^2/\text{s} \\
n_i &= 1.5 \times 10^{10} \, \text{cm}^{-3} \\
\tau_p = \tau_n &= 5 \times 10^{-7} \, \text{s} \\
\epsilon &= 11.7
\end{align*}
\]

**Solution**

The ideal reverse-saturation current density is given by

\[
J_s = eD_n \frac{n_{i0}}{L_n} + eD_p \frac{p_{i0}}{L_p}
\]