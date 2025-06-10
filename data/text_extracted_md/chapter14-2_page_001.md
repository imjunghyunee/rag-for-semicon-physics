# CHAPTER 14 Optical Devices

!Figure 14.18

**Figure 14.18** | Steady-state, photoinduced minority carrier concentrations and photocurrents in a “long” reverse-biased pn junction.

We can find the excess minority carrier hole concentration in the n region using the same type of analysis. Using the \( x' \) notation shown in Figure 14.17, we can write

\[
\delta p_n = G_T \tau_{p0} - (G_T \tau_{p0} + p_{n0}) e^{-x'/L_n}
\]

(14.38)

Equations (14.37) and (14.38) are plotted in Figure 14.18. We may note that the steady-state values far from the space charge region are the same as were given previously.

The gradient in the minority carrier concentrations will produce diffusion currents in the pn junction. The diffusion current density at \( x = 0 \) due to minority carrier electrons is

\[
J_{n1} = eD_n \left( \frac{d(\delta n_p)}{dx} \right)_{x=0} = eD_n \frac{d}{dx} [G_T \tau_{n0} - (G_T \tau_{n0} + n_{p0}) e^{-x/L_n}]_{x=0}
\]

\[
= \frac{eD_n}{L_n} (G_T \tau_{n0} + n_{p0})
\]

(14.39)

Equation (14.39) can be written as

\[
J_{n1} = eG_T L_n + \frac{eD_n n_{p0}}{L_n}
\]

(14.40)

The first term in Equation (14.40) is the steady-state photocurrent density while the second term is the ideal reverse saturation current density due to the minority carrier electrons.

The diffusion current density (in the x direction) at \( x' = 0 \) due to the minority carrier holes is

\[
J_{p1} = eG_T L_p + \frac{eD_p p_{n0}}{L_p}
\]

(14.41)