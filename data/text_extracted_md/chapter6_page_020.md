```markdown
majority carrier holes barely changes. However, we may have \(\delta n(0) \gg n_0\) and still satisfy the low-injection condition. The minority carrier concentration may change by many orders of magnitude.

----

### TEST YOUR UNDERSTANDING

**TYU 6.3** Excess electrons and holes are generated at the end of a silicon bar (\(x = 0\)). The silicon is doped with phosphorus atoms to a concentration of \(N_d = 10^{17} \, \text{cm}^{-3}\). The minority carrier lifetime is 1 μs, the electron diffusion coefficient is \(D_n = 25 \, \text{cm}^2/\text{s}\), and the hole diffusion coefficient is \(D_p = 10 \, \text{cm}^2/\text{s}\). If \(\delta p(0) = \delta n(0) = 10^5 \, \text{cm}^{-3}\), determine the steady-state electron and hole concentrations in the silicon for \(x > 0\).

**TYU 6.4** Using the parameters given in TYU 6.3, calculate the electron and hole diffusion current densities at \(x = 10 \, \mu\text{m}\).

----

The three previous examples, which applied the ambipolar transport equation to specific situations, assumed either a homogeneous or a steady-state condition; only the time variation or the spatial variation was considered. Now consider an example in which both the time and spatial dependence are considered in the same problem.

----

### Objective: Determine both the time dependence and spatial dependence of the excess carrier concentration.

Assume that a finite number of electron-hole pairs is generated instantaneously at time \(t = 0\) and at \(x = 0\), but assume \(g' = 0\) for \(t > 0\). Assume we have an n-type semiconductor with a constant applied electric field equal to \(E_0\), which is applied in the \(+x\) direction. Calculate the excess carrier concentration as a function of \(x\) and \(t\).

#### Solution

The one-dimensional ambipolar transport equation for the minority carrier holes can be written from Equation (6.56) as

\[
D_p \frac{\partial^2 (\delta p)}{\partial x^2} - \mu_p E_0 \frac{\partial (\delta p)}{\partial x} - \frac{\delta p}{\tau_{p0}} = \frac{\partial (\delta p)}{\partial t}
\]

(6.66)

The solution to this partial differential equation is of the form

\[
\delta p(x, t) = p'(x, t)e^{-t/\tau_{p0}}
\]

(6.67)

By substituting Equation (6.67) into Equation (6.66), we are left with the partial differential equation

\[
D_p \frac{\partial^2 p'(x, t)}{\partial x^2} - \mu_p E_0 \frac{\partial p'(x, t)}{\partial x} = \frac{\partial p'(x, t)}{\partial t}
\]

(6.68)
```