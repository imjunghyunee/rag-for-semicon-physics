# 11.3 Threshold Voltage Modifications

!Figure 11.14  
**Figure 11.14** | Cross section of a short n-channel MOSFET at flat band.

!Figure 11.15  
**Figure 11.15** | Charge sharing in the short-channel threshold voltage model.  
*(From Yau [261].)*

Using the geometrical approximation, the average bulk charge per unit area \( Q_B \) in the trapezoid is

\[
|Q_B| \cdot L = eN_Ax_{dT} \left( \frac{L + L'}{2} \right)
\]

(11.26)

From the geometry, we can show that

\[
\frac{L + L'}{2L} = \left[ 1 - \frac{r_j}{L} \left( \sqrt{1 + \frac{2x_{dT}}{r_j}} - 1 \right) \right]
\]

(11.27)

Then

\[
|Q_B| = eN_Ax_{dT} \left[ 1 - \frac{r_j}{L} \left( \sqrt{1 + \frac{2x_{dT}}{r_j}} - 1 \right) \right]
\]

(11.28)

Equation (11.28) is now used in place of \(|Q_{B0}(max)|\) in the expression for the threshold voltage.  
Since \(|Q_{B0}(max)| = eN_Ax_{dT}\), we can find \(\Delta V_T\) as

\[
\Delta V_T = -\frac{eN_Ax_{dT}}{C_{ox}} \left[ \frac{r_j}{L} \left( \sqrt{1 + \frac{2x_{dT}}{r_j}} - 1 \right) \right]
\]

(11.29)

where

\[
\Delta V_T = V_T(\text{short-channel}) - V_T(\text{long-channel})
\]

(11.30)

As the channel length decreases, the threshold voltage shifts in the negative direction so that an n-channel MOSFET shifts toward depletion mode.

----

**Objective:** Calculate the threshold voltage shift due to short-channel effects.

**Example 11.3**  
Consider an n-channel MOSFET with \( N_A = 3 \times 10^{16} \text{ cm}^{-3}, L = 1.0 \, \mu\text{m}, r_j = 0.3 \, \mu\text{m}, \text{ and } x_{dT} = 20 \, \text{nm} = 200 \, \text{Å}.**