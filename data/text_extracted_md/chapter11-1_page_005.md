# 11.1 Nonideal Effects

!Figure 11.5  
**Figure 11.5** | Cross-section of an n-channel MOSFET showing the channel length modulation effect.

For a one-sided n\(^+\)p junction, essentially all of the applied reverse-biased voltage is across the low-doped p region. The space charge width of the drain–substrate junction is approximately

\[
x_p = \sqrt{\frac{2\varepsilon_s}{eN_A}(\phi_{bp} + V_{DS})}
\]

(11.3)

However, the space charge region defined by \(\Delta L\), as shown in Figure 11.5, does not begin to form until \(V_{DS} > V_{DS(sat)}\). As a first approximation, we can write that \(\Delta L\) is the total space charge width minus the space charge width that exists when \(V_{DS} = V_{DS(sat)}\), or

\[
\Delta L = \sqrt{\frac{2\varepsilon_s}{eN_A} \left[ \phi_{bp} + V_{DS(sat)} + \Delta V_{DS} - \sqrt{\phi_{bp} + V_{DS(sat)}} \right]}
\]

(11.4)

where

\[
\Delta V_{DS} = V_{DS} - V_{DS(sat)}
\]

(11.5)

The applied drain-to-source voltage is \(V_{DS}\) and we are assuming that \(V_{DS} > V_{DS(sat)}\).

As a second approximation at determining \(\Delta L\), we can consider Figure 11.6 and revisit the one-dimensional Poisson’s equation. The electric field \(E_{sat}\) is the lateral electric field at the point where the inversion layer charge is pinched off. Neglecting any charges that exist due to current, we can write

\[
\frac{dE}{dx} = \frac{\rho(x)}{\varepsilon_s}
\]

(11.6)

where \(\rho(x) = -eN_A\), and is a constant for a uniformly doped substrate. Integrating Equation (11.6) and applying the boundary conditions give the electric field in the space charge region defined by \(\Delta L\):

\[
E = -\frac{eN_Ax}{\varepsilon_s} - E_{sat}
\]

(11.7)