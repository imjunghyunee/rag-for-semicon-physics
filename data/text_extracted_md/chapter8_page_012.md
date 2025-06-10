# 8.1 pn Junction Current

We can calculate the minority carrier hole diffusion current density at \( x = x_n \) from the relation

\[
J_p(x_n) = -eD_p \left. \frac{dp(x)}{dx} \right|_{x=x_n}
\]

(8.20)

Since we are assuming uniformly doped regions, the thermal-equilibrium carrier concentration is constant, so the hole diffusion current density may be written as

\[
J_p(x_n) = -eD_p \left. \frac{d\Delta p(x)}{dx} \right|_{x=x_n}
\]

(8.21)

Taking the derivative of Equation (8.14) and substituting into Equation (8.21), we obtain

\[
J_p(x_n) = \frac{eD_p n_{p0}}{L_p} \left[ \exp \left( \frac{eV_a}{kT} \right) - 1 \right]
\]

(8.22)

The hole current density for this forward-bias condition is in the \( +x \) direction, which is from the p to the n region.

Similarly, we may calculate the electron diffusion current density at \( x = -x_p \). This may be written as

\[
J_n(-x_p) = eD_n \left. \frac{d\Delta n(x)}{dx} \right|_{x=-x_p}
\]

(8.23)

Using Equation (8.15), we obtain

\[
J_n(-x_p) = \frac{eD_n n_{p0}}{L_n} \left[ \exp \left( \frac{eV_a}{kT} \right) - 1 \right]
\]

(8.24)

The electron current density is also in the \( +x \) direction.

An assumption we made at the beginning was that the individual electron and hole currents were continuous functions and constant through the space charge region. The total current is the sum of the electron and hole currents and is constant through the entire junction. Figure 8.7 again shows a plot of the magnitudes of these currents.

The total current density in the pn junction is then

\[
J = J_p(x_n) + J_n(-x_p) = \left[ \frac{eD_p n_{p0}}{L_p} + \frac{eD_n n_{p0}}{L_n} \right] \left[ \exp \left( \frac{eV_a}{kT} \right) - 1 \right]
\]

(8.25)

Equation (8.25) is the ideal current–voltage relationship of a pn junction.

We may define a parameter \( J_s \) as

\[
J_s = \frac{eD_p n_{p0}}{L_p} + \frac{eD_n n_{p0}}{L_n}
\]

(8.26)