# 14.3 Photodetectors

Similarly, the first term is the steady-state photocurrent density and the second term is the ideal reverse saturation current density.

The total steady-state diode photocurrent density for the long diode is now

\[
J_L = eG_LW + eG_LL_n + eG_LL_p = e(W + L_n + L_p)G_L
\]

(14.42)

Again note that the photocurrent is in the reverse-biased direction through the diode. The photocurrent given by Equation (14.42) is the result of assuming uniform generation of excess carriers throughout the structure, a long diode, and steady state.

The time response of the diffusion components of the photocurrent is relatively slow, since these currents are the results of the diffusion of minority carriers toward the depletion region. The diffusion components of photocurrent are referred to as the delayed photocurrent.

----

**Objective:** Calculate the steady-state photocurrent density in a reverse-biased, long pn diode.

**Example 14.5**

Consider a silicon pn diode at \( T = 300 \, \text{K} \) with the following parameters:

\[
\begin{align*}
N_a &= 10^{16} \, \text{cm}^{-3} \\
N_d &= 10^{16} \, \text{cm}^{-3} \\
D_n &= 25 \, \text{cm}^2/\text{s} \\
D_p &= 10 \, \text{cm}^2/\text{s} \\
\tau_{n0} &= 5 \times 10^{-7} \, \text{s} \\
\tau_{p0} &= 10^{-7} \, \text{s}
\end{align*}
\]

Assume that a reverse-biased voltage of \( V_R = 5 \) volts is applied and let \( G_L = 10^{21} \, \text{cm}^{-3}\text{s}^{-1} \).

## Solution

We may calculate various parameters as follows:

\[
L_n = \sqrt{D_n \tau_{n0}} = \sqrt{25 \times 5 \times 10^{-7}} = 35.4 \, \mu\text{m}
\]

\[
L_p = \sqrt{D_p \tau_{p0}} = \sqrt{10 \times 10^{-7}} = 10.0 \, \mu\text{m}
\]

\[
V_{bi} = V_t \ln \left( \frac{N_a N_d}{n_i^2} \right) = (0.0259) \ln \left( \frac{(10^{16})(10^{16})}{(1.5 \times 10^{10})^2} \right) = 0.695 \, \text{V}
\]

\[
W = \left[ \frac{2 \varepsilon_s}{e} \left( \frac{N_a + N_d}{N_a N_d} \right) (V_R + V_{bi}) \right]^{1/2}
\]

\[
= \left[ \frac{2(11.7)(8.85 \times 10^{-14})}{1.6 \times 10^{-19}} \left( \frac{2 \times 10^{16}}{(10^{16})(10^{16})} \right) (0.695 + 5) \right]^{1/2} = 1.21 \, \mu\text{m}
\]

Finally, the steady-state photocurrent density is

\[
J_L = e(W + L_n + L_p)G_L
\]

\[
= (1.6 \times 10^{-19})(1.21 + 35.4 + 10.0) \times 10^{-4}(10^{21}) = 0.75 \, \text{A/cm}^2
\]

## Comment

Again, keep in mind that this photocurrent is in the reverse-biased direction through the diode and is many orders of magnitude larger than the reverse-biased saturation current density in the pn junction diode.