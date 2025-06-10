```markdown
where \( G_{n0} \) and \( g'_n \) are the thermal-equilibrium electron and excess electron generation rates, respectively. The terms \( R_{n0} \) and \( R'_n \) are the thermal-equilibrium electron and excess electron recombination rates, respectively. For thermal equilibrium, we have that

\[
G_{n0} = R_{n0}
\]

(6.50)

so Equation (6.49) reduces to

\[
g - R = g'_n - R'_n = g'_n - \frac{\delta n}{\tau_n}
\]

(6.51)

where \( \tau_n \) is the excess minority carrier electron lifetime.

For the case of holes, we may write

\[
g - R = g_p - R_p = (G_{p0} + g'_p) - (R_{p0} + R'_p)
\]

(6.52)

where \( G_{p0} \) and \( g'_p \) are the thermal-equilibrium hole and excess hole generation rates, respectively. The terms \( R_{p0} \) and \( R'_p \) are the thermal-equilibrium hole and excess hole recombination rates, respectively. Again, for thermal equilibrium, we have that

\[
G_{p0} = R_{p0}
\]

(6.53)

so that Equation (6.52) reduces to

\[
g - R = g'_p - R'_p = g'_p - \frac{\delta p}{\tau_p}
\]

(6.54)

where \( \tau_p \) is the excess minority carrier hole lifetime.

The generation rate for excess electrons must equal the generation rate for excess holes. We may then define a generation rate for excess carriers as \( g' \), so that \( g'_n = g'_p = g' \). We also determined that the minority carrier lifetime is essentially a constant for low injection. Then, the term \( g - R \) in the ambipolar transport equation may be written in terms of the minority carrier parameters.

The ambipolar transport equation, given by Equation (6.39), for a p-type semiconductor under low injection then becomes

\[
D_n \frac{\partial^2 (\delta n)}{\partial x^2} - \mu_n E \frac{\partial (\delta n)}{\partial x} + g' - \frac{\delta n}{\tau_{n0}} = \frac{\partial (\delta n)}{\partial t}
\]

(6.55)

The parameter \( \delta n \) is the excess minority carrier electron concentration, the parameter \( \tau_{n0} \) is the minority carrier lifetime under low injection, and the other parameters are the usual minority carrier electron parameters.

Similarly, for an extrinsic n-type semiconductor under low injection, the ambipolar transport equation becomes

\[
D_p \frac{\partial^2 (\delta p)}{\partial x^2} - \mu_p E \frac{\partial (\delta p)}{\partial x} + g' - \frac{\delta p}{\tau_{p0}} = \frac{\partial (\delta p)}{\partial t}
\]

(6.56)

The parameter \( \delta p \) is the excess minority carrier hole concentration, the parameter \( \tau_{p0} \) is the minority carrier hole lifetime under low injection, and the other parameters are the usual minority carrier hole parameters.
```