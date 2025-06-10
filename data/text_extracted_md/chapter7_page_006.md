```markdown
!Figure 7.4

**Figure 7.4** The space charge density in a uniformly doped pn junction assuming the abrupt junction approximation.

### 7.2.2 Electric Field

An electric field is created in the depletion region by the separation of positive and negative space charge densities. Figure 7.4 shows the volume charge density distribution in the pn junction assuming uniform doping and assuming an abrupt junction approximation. We will assume that the space charge region abruptly ends in the n region at \( x = +x_n \), and abruptly ends in the p region at \( x = -x_p \) (\( x_p \) is a positive quantity).

The electric field is determined from Poisson’s equation, which, for a one-dimensional analysis, is

\[
\frac{d^2 \phi(x)}{dx^2} = -\frac{\rho(x)}{\epsilon_s} = -\frac{dE(x)}{dx}
\]

(7.11)

where \( \phi(x) \) is the electric potential, \( E(x) \) is the electric field, \( \rho(x) \) is the volume charge density, and \( \epsilon_s \) is the permittivity of the semiconductor. From Figure 7.4, the charge densities are

\[
\rho(x) = -eN_a \quad -x_p < x < 0
\]

(7.12a)

and

\[
\rho(x) = eN_d \quad 0 < x < x_n
\]

(7.12b)

The electric field in the p region is found by integrating Equation (7.11). We have

\[
E = \int \frac{\rho(x)}{\epsilon_s} \, dx = -\int \frac{eN_a}{\epsilon_s} \, dx = -\frac{eN_a x}{\epsilon_s} + C_1
\]

(7.13)

where \( C_1 \) is a constant of integration. The electric field is assumed to be zero in the neutral p region for \( x < -x_p \), since the currents are zero in thermal equilibrium. Since there are no surface charge densities within the pn junction structure, the electric
```