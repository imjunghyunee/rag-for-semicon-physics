To make the solution of Equations (6.29), (6.30), and (6.32) more tractable, we need to make some approximations. We can show that only a relatively small internal electric field is sufficient to keep the excess electrons and holes drifting and diffusing together. Hence, we can assume that

\[
|\mathbf{E}_{\text{int}}| \ll |\mathbf{E}_{\text{appl}}|
\]

(6.33)

However, the \(\nabla \cdot \mathbf{E}_{\text{int}}\) term may not be negligible. We will impose the condition of charge neutrality: We will assume that the excess electron concentration is just balanced by an equal excess hole concentration at any point in space and time. If this assumption were exactly true, there would be no induced internal electric field to keep the two sets of particles together. However, only a very small difference in the excess electron concentration and excess hole concentration will set up an internal E-field sufficient to keep the particles diffusing and drifting together. We can show that a 1 percent difference in \(\delta p\) and \(\delta n\), for example, will result in non-negligible values of the \(\nabla \cdot \mathbf{E} = \nabla \cdot \mathbf{E}_{\text{int}}\) term in Equations (6.29) and (6.30).

We can combine Equations (6.29) and (6.30) to eliminate the \(\nabla \cdot \mathbf{E}\) term. Considering Equations (6.1) and (6.4), we can define

\[
g_n = g_p = g
\]

(6.34)

and considering Equations (6.2) and (6.6), we can define

\[
R_n = \frac{n}{\tau_{n}} = R_p = \frac{p}{\tau_{p}} \equiv R
\]

(6.35)

The lifetimes in Equation (6.35) include the thermal-equilibrium carrier lifetimes and the excess carrier lifetimes. If we impose the charge neutrality condition, then \(\delta n \approx \delta p\). We will denote both the excess electron and excess hole concentrations in Equations (6.29) and (6.30) by \(\delta n\). We may then rewrite Equations (6.29) and (6.30) as

\[
D_n \frac{\partial^2 (\delta n)}{\partial x^2} - \mu_p \left[ E \frac{\partial (\delta n)}{\partial x} + p \frac{\partial E}{\partial x} \right] + g - R = \frac{\partial (\delta n)}{\partial t}
\]

(6.36)

and

\[
D_p \frac{\partial^2 (\delta n)}{\partial x^2} + \mu_e \left[ E \frac{\partial (\delta n)}{\partial x} + n \frac{\partial E}{\partial x} \right] + g - R = \frac{\partial (\delta n)}{\partial t}
\]

(6.37)

If we multiply Equation (6.36) by \(\mu_n\), multiply Equation (6.37) by \(\mu_p\), and add the two equations, the \(\nabla \cdot \mathbf{E} = \partial E / \partial x\) term will be eliminated. The result of this addition gives

\[
(\mu_n n D_n + \mu_p D_p) \frac{\partial^2 (\delta n)}{\partial x^2} + (\mu_n \mu_p)(p - n) E \frac{\partial (\delta n)}{\partial x}
\]

\[
+ (\mu_n n + \mu_p p)(g - R) = (\mu_n + \mu_p) \frac{\partial (\delta n)}{\partial t}
\]

(6.38)

If we divide Equation (6.38) by the term \((\mu_n + \mu_p)\), this equation becomes

\[
D' \frac{\partial^2 (\delta n)}{\partial x^2} + \mu' E \frac{\partial (\delta n)}{\partial x} + g - R = \frac{\partial (\delta n)}{\partial t}
\]

(6.39)