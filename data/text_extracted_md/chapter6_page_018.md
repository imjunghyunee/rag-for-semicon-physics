# Objective

Determine the steady-state spatial dependence of the excess carrier concentration.

Consider a p-type semiconductor that is homogeneous and infinite in extent. Assume a zero applied electric field. For a one-dimensional crystal, assume that excess carriers are being generated at \( x = 0 \) only, as indicated in Figure 6.6. The excess carriers being generated at \( x = 0 \) will begin diffusing in both the \( +x \) and \( -x \) directions. Calculate the steady-state excess carrier concentration as a function of \( x \).

## Solution

The ambipolar transport equation for excess minority carrier electrons is given by Equation (6.55), and is written as

\[
D_n \frac{d^2 (\delta n)}{dx^2} + \mu_n E \frac{d(\delta n)}{dx} + g' - \frac{\delta n}{\tau_0} = \frac{\partial (\delta n)}{\partial t}
\]

From our assumptions, we have \( E = 0 \), \( g' = 0 \) for \( x \neq 0 \), and \( \partial (\delta n)/\partial t = 0 \) for steady state. Assuming a one-dimensional crystal, Equation (6.55) reduces to

\[
D_n \frac{d^2 (\delta n)}{dx^2} - \frac{\delta n}{\tau_0} = 0
\]

(6.62)

Dividing by the diffusion coefficient, Equation (6.62) may be written as

\[
\frac{d^2 (\delta n)}{dx^2} - \frac{\delta n}{D_n \tau_0} = \frac{d^2 (\delta n)}{dx^2} - \frac{\delta n}{L_n^2} = 0
\]

(6.63)

where we have defined \( L_n^2 = D_n \tau_n \). The parameter \( L_n \) has the unit of length and is called the minority carrier electron diffusion length. The general solution to Equation (6.63) is

\[
\delta n(x) = Ae^{-x/L_n} + Be^{x/L_n}
\]

(6.64)

As the minority carrier electrons diffuse away from \( x = 0 \), they will recombine with the majority carrier holes. The minority carrier electron concentration will then decay toward zero at both \( x = +\infty \) and \( x = -\infty \). These boundary conditions mean that \( B = 0 \) for \( x > 0 \) and \( A = 0 \) for \( x < 0 \). The solution to Equation (6.63) may then be written as

\[
\delta n(x) = \delta n(0)e^{-x/L_n}, \quad x \geq 0
\]

(6.65a)

!Figure 6.6

**Figure 6.6** Steady-state generation rate at \( x = 0 \).