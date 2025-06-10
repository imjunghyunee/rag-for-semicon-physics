## 6.6 Surface Effects

The solution to Equation (6.107) is of the form

\[
\delta p(x) = g' \tau_{p0} + Ae^{x/L_p} + Be^{-x/L_p}
\]

(6.109)

As \( x \to +\infty \), \(\delta p(x) = \delta p = g' \tau_{p0} = 10^{14} \, \text{cm}^{-3}\), which implies that \( A = 0 \). At \( x = 0 \), we have

\[
\delta p(0) = \delta p_s = 10^{14} + B = 10^{13} \, \text{cm}^{-3}
\]

so that \( B = -9 \times 10^{13} \). The entire solution for the minority carrier hole concentration as a function of distance from the surface is

\[
\delta p(x) = 10^{14} (1 - 0.9 e^{-x/L_p})
\]

where

\[
L_p = \sqrt{D_p \tau_{p0}} = \sqrt{(10)(10^{-6})} = 31.6 \, \mu \text{m}
\]

### Comment

The excess carrier concentration is smaller at the surface than in the bulk.

### EXERCISE PROBLEM

**Ex 6.9** (a) Repeat Example 6.9 for the case when \(\tau_{p0} = 0\). (b) What is the excess hole concentration at \( x = 0 \)? (c) For this particular case, what is the recombination rate of excess carriers at the surface?

\[
[\omega = x] (y) 0 = (0)\delta p (q) \frac{q}{\tau_{p0}} - 1) \psi_{L_p} \beta = (\chi \delta p) \text{SU}
\]

## 6.6.2 Surface Recombination Velocity

A gradient in the excess carrier concentration exists near the surface as shown in Figure 6.18; excess carriers from the bulk region diffuse toward the surface where they recombine. This diffusion toward the surface can be described by

\[
-D_p \left[ \hat{n} \cdot \left( \frac{d \delta p}{d x} \right) \right]_{\text{surf}} = s \delta p_{\text{surf}}
\]

(6.110)

where each side of the equation is evaluated at the surface. The parameter \(\hat{n}\) is the unit outward vector normal to the surface. Using the geometry of Figure 6.18, \(d(\delta p)/dx\) is a positive quantity and \(\hat{n}\) is negative, so that the parameter \(s\) is a positive quantity.

A dimensional analysis of Equation (6.110) shows that the parameter \(s\) has units of cm/s, or velocity. The parameter \(s\) is called the **surface recombination velocity**. If the excess concentrations at the surface and in the bulk region were equal, then the gradient term would be zero and the surface recombination velocity would be zero. As the excess concentration at the surface becomes smaller, the gradient term becomes larger, and the surface recombination velocity increases. The surface recombination velocity gives some indication of the surface characteristics as compared with the bulk region.