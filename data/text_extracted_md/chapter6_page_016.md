# Solution

For the n-type semiconductor, we need to consider the ambipolar transport equation for the minority carrier holes, which is given by Equation (6.56). The equation is

\[
D_p \frac{\partial^2 (\delta p)}{\partial x^2} - \mu_p E \frac{\partial (\delta p)}{\partial x} + g' - \frac{\delta p}{\tau_{p0}} = \frac{\partial (\delta p)}{\partial t}
\]

We are assuming a uniform concentration of excess holes so that \(\partial^2 (\delta p)/\partial x^2 = \partial (\delta p)/\partial x = 0\). For \(t > 0\), we are also assuming that \(g' = 0\). Equation (6.56) reduces to

\[
\frac{\partial (\delta p)}{\partial t} = -\frac{\delta p}{\tau_{p0}}
\]

(6.57)

Since there is no spatial variation, the total time derivative may be used. At low injection, the minority carrier hole lifetime, \(\tau_{p0}\), is a constant. The left side of Equation (6.57) is the time rate of change of \(\delta p\) and the right side of the equation is the recombination rate. The solution to Equation (6.57) is

\[
\delta p(t) = \delta p(0)e^{-t/\tau_{p0}}
\]

(6.58)

where \(\delta p(0)\) is the uniform concentration of excess carriers that exists at time \(t = 0\). The concentration of excess holes decays exponentially with time, with a time constant equal to the minority carrier hole lifetime.

From the charge-neutrality condition, we have that \(\delta n = \delta p\), so the excess electron concentration is given by

\[
\delta n(t) = \delta p(t) = \delta p(0)e^{-t/\tau_{p0}}
\]

(6.59)

# Comment

The excess electrons and holes recombine at the rate determined by the excess minority carrier hole lifetime in the n-type semiconductor.

# Exercise Problem

**Ex. 6.2** Consider n-type GaAs doped at \(N_d = 10^{16} \text{ cm}^{-3}\). Assume that \(10^{14}\) electron–hole pairs have been uniformly created per cm\(^3\) at \(t = 0\), and assume the minority carrier hole lifetime is \(\tau_{p0} = 50 \text{ ns}\). Determine the time at which the minority carrier hole concentration reaches (a) 1/10 of its initial value and (b) 10% of its initial value.

\[ \text{[SI 1 = (a) 50 ns = 1 (b) 50 ns]} \]

----

# Objective

Determine the time dependence of excess carriers in reaching a steady-state condition.

Again consider an infinitely large, homogeneous n-type semiconductor with a zero applied electric field. Assume that, for \(t < 0\), the semiconductor is in thermal equilibrium and that, for \(t \geq 0\), a uniform generation rate exists in the crystal. Calculate the excess carrier concentration as a function of time assuming the condition of low injection.

**Example 6.3**