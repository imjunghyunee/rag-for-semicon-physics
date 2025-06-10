# Metal–Oxide–Semiconductor Field-Effect Transistor: Additional Concepts

!Figure 11.6  
**Figure 11.6** Expanded view of cross section near the drain terminal of an n-channel MOSFET showing the channel length modulation effect.

The potential in this region is

\[
\phi(x) = -\int E \, dx = \frac{eN_s x^2}{2\epsilon} + E_{\text{sat}} x + C_1
\]

(11.8)

where \( C_1 \) is a constant of integration. The boundary conditions are \(\phi(x = 0) = V_{DS(\text{sat})}\) and \(\phi(x = \Delta L) = V_{DS}\). Substituting these boundary conditions into Equation (11.8), we obtain

\[
V_{DS} = \frac{eN_s (\Delta L)^2}{2\epsilon} + E_{\text{sat}} (\Delta L) + V_{DS(\text{sat})}
\]

(11.9)

Solving for \(\Delta L\), we can write

\[
\Delta L = \sqrt{\frac{2\epsilon}{eN_s} \left[ \sqrt{\phi_{\text{sat}} + [V_{DS} - V_{DS(\text{sat})}]} - \sqrt{\phi_{\text{sat}}} \right]}
\]

(11.10)

where

\[
\phi_{\text{sat}} = \frac{2\epsilon}{eN_s} \cdot \left( \frac{E_{\text{sat}}}{2} \right)^2
\]

In general, the value of \( E_{\text{sat}} \) is in the range \( 10^4 < E_{\text{sat}} < 2 \times 10^5 \) V/cm.

Other models used to determine \(\Delta L\) include the negative charges due to the drain current and also include two-dimensional effects. These models are not considered here.

Since the drain current is inversely proportional to the channel length, we may write

\[
I_D = \left[ \frac{L}{L - \Delta L} \right] I_{D0}
\]

(11.11)

where \( I_D \) is the actual drain current and \( I_{D0} \) is the ideal drain current. Since \(\Delta L\) is a function of \( V_{DS} \), \( I_D \) is now also a function of \( V_{DS} \) even though the transistor is biased in the saturation region.