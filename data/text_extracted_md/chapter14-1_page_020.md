# 14.3 Photodetectors

The space charge region. If we assume that the saturation drift velocity is \(10^7 \, \text{cm/s}\) and the depletion width is \(2 \, \mu\text{m}\), the transit time is \(\tau_t = 20 \, \text{ps}\). The ideal modulating frequency has a period of \(2\tau_t\), so the frequency is \(f = 25 \, \text{GHz}\). This frequency response is substantially higher than that of photoconductors.

Excess carriers are also generated within the neutral n and p regions of the diode. The excess minority carrier electron distribution in the p region is found from the ambipolar transport equation, which is

\[
D_n \frac{\partial^2 (\delta n_p)}{\partial x^2} + G_t - \frac{\delta n_p}{\tau_{n0}} = \frac{\partial (\delta n_p)}{\partial t}
\]

(14.29)

We will assume that the E-field is zero in the neutral regions. In steady state, \(\partial (\delta n_p)/\partial t = 0\), so that Equation (14.29) can be written as

\[
\frac{d^2 (\delta n_p)}{dx^2} - \frac{\delta n_p}{L_n^2} = -\frac{G_t}{D_n}
\]

(14.30)

where \(L_n^2 = D_n \tau_{n0}\).

The solution to Equation (14.30) can be found as the sum of the homogeneous and particular solutions. The homogeneous solution is found from the equation

\[
\frac{d^2 (\delta n_{ph})}{dx^2} - \frac{\delta n_{ph}}{L_n^2} = 0
\]

(14.31)

where \(\delta n_{ph}\) is the homogeneous solution and is given by

\[
\delta n_{ph} = Ae^{-x/L_n} + Be^{x/L_n}, \quad (x \geq 0)
\]

(14.32)

One boundary condition is that \(\delta n_{ph}\) must remain finite, which implies that \(B = 0\) for the "long" diode.

The particular solution is found from

\[
-\frac{\delta n_{pp}}{L_n^2} = -\frac{G_t}{D_n}
\]

(14.33)

which yields

\[
\delta n_{pp} = \frac{G_t L_n^2}{D_n} = \frac{G_t (D_n \tau_{n0})}{D_n} = G_t \tau_{n0}
\]

(14.34)

The total steady-state solution for the excess minority carrier electron concentration in the p region is then

\[
\delta n_p = Ae^{-x/L_n} + G_t \tau_{n0}
\]

(14.35)

The total electron concentration is zero at \(x = 0\) for the reverse-biased junction. The excess electron concentration \(x = 0\) is then

\[
\delta n_p (x = 0) = -n_{p0}
\]

(14.36)

Using the boundary condition from Equation (14.36), the electron concentration given by Equation (14.35) becomes

\[
\delta n_p = G_t \tau_{n0} - (G_t \tau_{n0} + n_{p0}) e^{-x/L_n}
\]

(14.37)