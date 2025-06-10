## 6.2 Characteristics of Excess Carriers

This equation is a Taylor expansion of \( F^+_p(x + dx) \), where the differential length \( dx \) is small, so that only the first two terms in the expansion are significant. The net increase in the number of holes per unit time within the differential volume element due to the \( x \)-component of hole flux is given by

\[
\frac{\partial p}{\partial t} dx \, dy \, dz = [F^+_p(x) - F^+_p(x + dx)] dy \, dz = -\frac{\partial F^+_p}{\partial x} dx \, dy \, dz
\]

(6.16)

If \( F^+_p(x) > F^+_p(x + dx) \), for example, there will be a net increase in the number of holes in the differential volume element with time. If we generalize to a three-dimensional hole flux, then the right side of Equation (6.16) may be written as \(-\nabla \cdot F^+_p \, dx \, dy \, dz\), where \(\nabla \cdot F^+_p\) is the divergence of the flux vector. We will limit ourselves to a one-dimensional analysis.

The generation rate and recombination rate of holes will also affect the hole concentration in the differential volume. The net increase in the number of holes per unit time in the differential volume element is then given by

\[
\frac{\partial p}{\partial t} dx \, dy \, dz = -\frac{\partial F^+_p}{\partial x} dx \, dy \, dz + g_p \, dx \, dy \, dz - \frac{p}{\tau_{p0}} dx \, dy \, dz
\]

(6.17)

where \( p \) is the density of holes. The first term on the right side of Equation (6.17) is the increase in the number of holes per unit time due to the hole flux, the second term is the increase in the number of holes per unit time due to the generation of holes, and the last term is the decrease in the number of holes per unit time due to the recombination of holes. The recombination rate for holes is given by \( p/\tau_{p0} \), where \(\tau_{p0}\) includes the thermal-equilibrium carrier lifetime and the excess carrier lifetime.

If we divide both sides of Equation (6.17) by the differential volume \( dx \, dy \, dz \), the net increase in the hole concentration per unit time is

\[
\frac{\partial p}{\partial t} = -\frac{\partial F^+_p}{\partial x} + g_p - \frac{p}{\tau_{p0}}
\]

(6.18)

Equation (6.18) is known as the continuity equation for holes.

Similarly, the one-dimensional continuity equation for electrons is given by

\[
\frac{\partial n}{\partial t} = -\frac{\partial F^-_n}{\partial x} + g_n - \frac{n}{\tau_{n0}}
\]

(6.19)

where \( F^-_n \) is the electron-particle flow, or flux, also given in units of number of electrons/cm\(^2\)-s.

### 6.2.2 Time-Dependent Diffusion Equations

In Chapter 5, we derived the hole and electron current densities, which are given, in one dimension, by

\[
J_p = e\mu_p pE - eD_p \frac{\partial p}{\partial x}
\]

(6.20)

and

\[
J_n = e\mu_n nE + eD_n \frac{\partial n}{\partial x}
\]

(6.21)