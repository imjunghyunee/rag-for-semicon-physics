## 7.2 Zero Applied Bias

Field is a continuous function. The constant of integration is determined by setting \( E = 0 \) at \( x = -x_p \). The electric field in the p region is then given by

\[
E = -\frac{eN_a}{\varepsilon_s}(x + x_p) \quad -x_p \leq x \leq 0
\]

(7.14)

In the n region, the electric field is determined from

\[
E = \int \frac{eN_d}{\varepsilon_s} \, dx = \frac{eN_d}{\varepsilon_s} x + C_2
\]

(7.15)

where \( C_2 \) is again a constant of integration and is determined by setting \( E = 0 \) at \( x = x_n \), since the E-field is assumed to be zero in the n region and is a continuous function. Then

\[
E = -\frac{eN_d}{\varepsilon_s}(x_n - x) \quad 0 \leq x \leq x_n
\]

(7.16)

The electric field is also continuous at the metallurgical junction, or at \( x = 0 \). Setting Equations (7.14) and (7.16) equal to each other at \( x = 0 \) gives

\[
N_a x_p = N_d x_n
\]

(7.17)

Equation (7.17) states that the number of negative charges per unit area in the p region is equal to the number of positive charges per unit area in the n region.

Figure 7.5 is a plot of the electric field in the depletion region. The electric field direction is from the n to the p region, or in the negative x direction for this geometry. For the uniformly doped pn junction, the E-field is a linear function of distance through the junction, and the maximum (magnitude) electric field occurs at the metallurgical junction. An electric field exists in the depletion region even when no voltage is applied between the p and n regions.

The potential in the junction is found by integrating the electric field. In the p region then, we have

\[
\phi(x) = -\int E(x) dx = \int \frac{eN_a}{\varepsilon_s}(x + x_p) dx
\]

(7.18)

!Figure 7.5

**Figure 7.5** Electric field in the space charge region of a uniformly doped pn junction.