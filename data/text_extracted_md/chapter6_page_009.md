# CHAPTER 6 Non-equilibrium Excess Carriers in Semiconductors

If we divide the hole current density by \( (+e) \) and the electron current density by \( (-e) \), we obtain each particle flux. These equations become

\[
\frac{J_p}{(+e)} = F^+ = \mu_p pE - D_p \frac{\partial p}{\partial x}
\]
(6.22)

and

\[
\frac{J_n}{(-e)} = F^- = -\mu_n nE - D_n \frac{\partial n}{\partial x}
\]
(6.23)

Taking the divergence of Equations (6.22) and (6.23), and substituting back into the continuity equations of (6.18) and (6.19), we obtain

\[
\frac{\partial p}{\partial t} = -\mu_p \frac{\partial (pE)}{\partial x} + D_p \frac{\partial^2 p}{\partial x^2} + g_p - \frac{p}{\tau_p}
\]
(6.24)

and

\[
\frac{\partial n}{\partial t} = \mu_n \frac{\partial (nE)}{\partial x} + D_n \frac{\partial^2 n}{\partial x^2} + g_n - \frac{n}{\tau_n}
\]
(6.25)

Keeping in mind that we are limiting ourselves to a one-dimensional analysis, we can expand the derivative of the product as

\[
\frac{\partial (pE)}{\partial x} = E \frac{\partial p}{\partial x} + p \frac{\partial E}{\partial x}
\]
(6.26)

In a more generalized three-dimensional analysis, Equation (6.26) would have to be replaced by a vector identity. Equations (6.24) and (6.25) can be written in the form

\[
D_p \frac{\partial^2 p}{\partial x^2} - \mu_p \left( E \frac{\partial p}{\partial x} + p \frac{\partial E}{\partial x} \right) + g_p - \frac{p}{\tau_p} = \frac{\partial p}{\partial t}
\]
(6.27)

and

\[
D_n \frac{\partial^2 n}{\partial x^2} + \mu_n \left( E \frac{\partial n}{\partial x} + n \frac{\partial E}{\partial x} \right) + g_n - \frac{n}{\tau_n} = \frac{\partial n}{\partial t}
\]
(6.28)

Equations (6.27) and (6.28) are the time-dependent diffusion equations for holes and electrons, respectively. Since both the hole concentration \( p \) and the electron concentration \( n \) contain the excess concentrations, Equations (6.27) and (6.28) describe the space and time behavior of the excess carriers.

The hole and electron concentrations are functions of both the thermal equilibrium and the excess values, which are given in Equations (6.5a) and (6.5b). The thermal-equilibrium concentrations, \( n_0 \) and \( p_0 \), are not functions of time. For the special case of a homogeneous semiconductor, \( n_0 \) and \( p_0 \) are also independent of the space coordinates. Equations (6.27) and (6.28) may then be written in the form

\[
D_p \frac{\partial^2 (\delta p)}{\partial x^2} - \mu_p \left( E \frac{\partial (\delta p)}{\partial x} + p \frac{\partial E}{\partial x} \right) + g_p - \frac{p}{\tau_p} = \frac{\partial (\delta p)}{\partial t}
\]
(6.29)

and

\[
D_n \frac{\partial^2 (\delta n)}{\partial x^2} + \mu_n \left( E \frac{\partial (\delta n)}{\partial x} + n \frac{\partial E}{\partial x} \right) + g_n - \frac{n}{\tau_n} = \frac{\partial (\delta n)}{\partial t}
\]
(6.30)