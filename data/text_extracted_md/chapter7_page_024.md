# CHAPTER 7 The pn Junction

doped semiconductor is not exactly zero, but the magnitude of this field is small, so setting \( E = 0 \) in the bulk regions is still a good approximation.

The potential is again found by integrating the electric field as

\[
\phi(x) = - \int E \, dx
\]

(7.65)

If we arbitrarily set \(\phi = 0\) at \(x = -x_0\), then the potential through the junction is

\[
\phi(x) = -\frac{ea}{2\epsilon_s} \left( \frac{x^3}{3} - x_0^2 x \right) + \frac{ea}{3\epsilon_s} x_0^3
\]

(7.66)

The magnitude of the potential at \(x = +x_0\) will equal the built-in potential barrier for this function. We then have that

\[
\phi(x_0) = \frac{2}{3} \cdot \frac{ea x_0^3}{\epsilon_s} = V_{bi}
\]

(7.67)

Another expression for the built-in potential barrier for a linearly graded junction can be approximated from the expression used for a uniformly doped junction. We can write

\[
V_{bi} = V_t \ln \left[ \frac{N_d(x_0) N_a(-x_0)}{n_i^2} \right]
\]

(7.68)

where \(N_d(x_0)\) and \(N_a(-x_0)\) are the doping concentrations at the edges of the space charge region. We can relate these doping concentrations to the gradient, so that

\[
N_d(x_0) = ax_0
\]

(7.69a)

and

\[
N_a(-x_0) = ax_0
\]

(7.69b)

Then the built-in potential barrier for the linearly graded junction becomes

\[
V_{bi} = V_t \ln \left( \frac{(ax_0)^2}{n_i^2} \right)
\]

(7.70)

There may be situations in which the doping gradient is not the same on either side of the junction, but we will not consider that condition here.

If a reverse-biased voltage is applied to the junction, the potential barrier increases. The built-in potential barrier \(V_{bi}\) in the above equations is then replaced by the total potential barrier \(V_{bi} + V_R\). Solving for \(x_0\) from Equation (7.67) and using the total potential barrier, we obtain

\[
x_0 = \left[ \frac{3}{2} \cdot \frac{\epsilon_s}{ea} (V_{bi} + V_R) \right]^{1/3}
\]

(7.71)

The junction capacitance per unit area can be determined by the same method that we used for the uniformly doped junction. Figure 7.18 shows the differential charge