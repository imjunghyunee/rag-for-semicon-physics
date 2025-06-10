# 14.2 Solar Cells

!I-V characteristics of a pn junction solar cell.

**Figure 14.7** I–V characteristics of a pn junction solar cell.

where the ideal diode equation has been used. As the diode becomes forward biased, the magnitude of the electric field in the space charge region decreases, but does not go to zero or change direction. The photocurrent is always in the reverse-biased direction and the net solar cell current is also always in the reverse-biased direction.

There are two limiting cases of interest. The short-circuit condition occurs when \( R = 0 \) so that \( V = 0 \). The current in this case is referred to as the **short-circuit current**, or

\[
I = I_{sc} = I_L
\]

(14.8)

The second limiting case is the open-circuit condition and occurs when \( R \rightarrow \infty \). The net current is zero and the voltage produced is the **open-circuit voltage**. The photocurrent is just balanced by the forward-biased junction current, so we have

\[
I = 0 = I_L - I_s \left[ \exp \left( \frac{eV}{kT} \right) - 1 \right]
\]

(14.9)

We can find the open circuit voltage \( V_{oc} \) as

\[
V_{oc} = V_t \ln \left( 1 + \frac{I_L}{I_s} \right)
\]

(14.10)

A plot of the diode current \( I \) as a function of the diode voltage \( V \) from Equation (14.7) is shown in Figure 14.7. We may note the short-circuit current and open-circuit voltage points on the figure.

----

**Objective:** Calculate the open-circuit voltage of a silicon pn junction solar cell.

**Example 14.3**

Consider a silicon pn junction at \( T = 300 \, \text{K} \) with the following parameters:

\[
\begin{align*}
N_d &= 5 \times 10^{18} \, \text{cm}^{-3} \\
N_a &= 10^{16} \, \text{cm}^{-3} \\
D_n &= 25 \, \text{cm}^2/\text{s} \\
D_p &= 10 \, \text{cm}^2/\text{s} \\
\tau_{n0} &= 5 \times 10^{-7} \, \text{s} \\
\tau_{p0} &= 10^{-7} \, \text{s}
\end{align*}
\]

Let the photocurrent density be \( J_L = I_L/A = 15 \, \text{mA/cm}^2 \).