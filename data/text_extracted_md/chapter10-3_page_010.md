# Fundamentals of the Metal–Oxide–Semiconductor Field-Effect Transistor

If \( V_{SB} = 0 \), threshold is defined as the condition when \( \phi_s = 2\phi_{fp} \) as we discussed previously and as shown in Figure 10.50b. When \( V_{SB} > 0 \) the surface will still try to invert when \( \phi_s = 2\phi_{fp} \). However, these electrons are at a higher potential energy than are the electrons in the source. The newly created electrons will move laterally and flow out of the source terminal. When \( \phi_s = 2\phi_{fp} + V_{SB} \), the surface reaches an equilibrium inversion condition. The energy-band diagram for this condition is shown in Figure 10.50c. The curve represented as \( E_F \) is the Fermi level from the p substrate through the reverse-biased source–substrate junction to the source contact.

The space charge region width under the oxide increases from the original \( x_{dT} \) value when a reverse-biased source–substrate junction voltage is applied. With an applied \( V_{SB} > 0 \), there is more charge associated with this region. Considering the charge neutrality condition through the MOS structure, the positive charge on the top metal gate must increase to compensate for the increased negative space charge in order to reach the threshold inversion point. So when \( V_{SB} > 0 \), the threshold voltage of the n-channel MOSFET increases.

When \( V_{SB} = 0 \), we had

\[
Q'_{SD} (\text{max}) = -eN_ax_{dT} = -\sqrt{2\epsilon_{ox} N_a(2\phi_{fp})}
\]

(10.78)

When \( V_{SB} > 0 \), the space charge width increases and we now have

\[
Q'_{SD} = -eN_ax_d = -\sqrt{2\epsilon_{ox} N_a(2\phi_{fp} + V_{SB})}
\]

(10.79)

The change in the space charge density is then

\[
\Delta Q'_{SD} = -\sqrt{2\epsilon_{ox} N_a} \left[ \sqrt{2\phi_{fp} + V_{SB}} - \sqrt{2\phi_{fp}} \right]
\]

(10.80)

To reach the threshold condition, the applied gate voltage must be increased. The change in threshold voltage can be written as

\[
\Delta V_T = -\frac{\Delta Q'_{SD}}{C_{ox}} = \frac{\sqrt{2\epsilon_{ox} N_a}}{C_{ox}} \left[ \sqrt{2\phi_{fp} + V_{SB}} - \sqrt{2\phi_{fp}} \right]
\]

(10.81)

where \( \Delta V_T = V_T(V_{SB} > 0) - V_T(V_{SB} = 0) \). We may note that \( V_{SB} \) must always be positive so that, for the n-channel device, \( \Delta V_T \) is always positive. The threshold voltage of the n-channel MOSFET will increase as a function of the source–substrate junction voltage.

From Equation (10.81), we may define

\[
\gamma = \frac{\sqrt{2\epsilon_{ox} N_a}}{C_{ox}}
\]

(10.82)

where \( \gamma \) is defined as the body-effect coefficient. Equation (10.81) may then be written as

\[
\Delta V_T = \gamma \left[ \sqrt{2\phi_{fp} + V_{SB}} - \sqrt{2\phi_{fp}} \right]
\]

(10.83)