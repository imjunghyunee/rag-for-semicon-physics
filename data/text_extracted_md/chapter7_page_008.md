or

\[
\phi(x) = \frac{eN_a}{\varepsilon_s} \left( \frac{x^2}{2} + x_p \cdot x \right) + C'_1
\]

(7.19)

where \( C'_1 \) is again a constant of integration. The potential difference through the pn junction is the important parameter, rather than the absolute potential, so we may arbitrarily set the potential equal to zero at \( x = -x_p \). The constant of integration is then found as

\[
C'_1 = \frac{eN_a}{2\varepsilon_s} x_p^2
\]

(7.20)

so that the potential in the p region can now be written as

\[
\phi(x) = \frac{eN_a}{2\varepsilon_s} (x + x_p)^2 \quad (-x_p \leq x \leq 0)
\]

(7.21)

The potential in the n region is determined by integrating the electric field in the n region, or

\[
\phi(x) = \int \frac{eN_d}{\varepsilon_s} (x_n - x) dx
\]

(7.22)

Then

\[
\phi(x) = \frac{eN_d}{\varepsilon_s} \left( x \cdot x_n - \frac{x^2}{2} \right) + C'_2
\]

(7.23)

where \( C'_2 \) is another constant of integration. The potential is a continuous function, so setting Equation (7.21) equal to Equation (7.23) at the metallurgical junction, or at \( x = 0 \), gives

\[
C'_2 = \frac{eN_a}{2\varepsilon_s} x_p^2
\]

(7.24)

The potential in the n region can thus be written as

\[
\phi(x) = \frac{eN_d}{\varepsilon_s} \left( x \cdot x_n - \frac{x^2}{2} \right) + \frac{eN_a}{2\varepsilon_s} x_p^2 \quad (0 \leq x \leq x_n)
\]

(7.25)

Figure 7.6 is a plot of the potential through the junction and shows the quadratic dependence on distance. The magnitude of the potential at \( x = x_n \) is equal to the built-in potential barrier. Then from Equation (7.25), we have

\[
V_{bi} = |\phi(x = x_n)| = \frac{e}{2\varepsilon_s} (N_ax_p^2 + N_dx_n^2)
\]

(7.26)

The potential energy of an electron is given by \( E = -e\phi \), which means that the electron potential energy also varies as a quadratic function of distance through the space charge region. The quadratic dependence on distance was shown in the energy-band diagram of Figure 7.3, although we did not explicitly know the shape of the curve at that time.