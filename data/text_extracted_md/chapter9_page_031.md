## 9.3 Heterojunctions

The total built-in potential barrier is

\[
V_{bi} = V_{bi,n} + V_{bi,p} = \frac{eN_A x_n^2}{2\epsilon_e} + \frac{eN_D x_p^2}{2\epsilon_p}
\]

(9.46)

If we solve for \(x_n\), for example, from Equation (9.42b) and substitute into Equation (9.46), we can solve for \(x_n\) as

\[
x_n = \left[ \frac{2\epsilon_e \epsilon_p N_A V_{bi}}{eN_A(\epsilon_e N_A + \epsilon_p N_D)} \right]^{1/2}
\]

(9.47a)

We can also find

\[
x_p = \left[ \frac{2\epsilon_e \epsilon_p N_D V_{bi}}{eN_D(\epsilon_e N_A + \epsilon_p N_D)} \right]^{1/2}
\]

(9.47b)

The total depletion width is found to be

\[
W = x_n + x_p = \left[ \frac{2\epsilon_e \epsilon_p (N_A + N_D)^2 V_{bi}}{eN_A N_D(\epsilon_e N_A + \epsilon_p N_D)} \right]^{1/2}
\]

(9.48)

If a reverse-biased voltage is applied across the heterojunction, the same equations apply if \(V_{bi}\) is replaced by \(V_{bi} + V_R\). Similarly, if a forward bias is applied, the same equations also apply if \(V_{bi}\) is replaced by \(V_{bi} - V_a\). As explained earlier, \(V_R\) is the magnitude of the reverse-biased voltage and \(V_a\) is the magnitude of the forward-bias voltage.

As in the case of a homojunction, a change in depletion width with a change in junction voltage yields a junction capacitance. We can find for the nP junction

\[
C_j' = \left[ \frac{eN_A N_D \epsilon_e \epsilon_p}{2(\epsilon_e N_A + \epsilon_p N_D)(V_{bi} + V_R)} \right]^{1/2} \quad \text{(F/cm}^2\text{)}
\]

(9.49)

A plot of \(1/(C_j')^2\) versus \(V_R\) again yields a straight line. The extrapolation of this plot of \(1/(C_j')^2 = 0\) is used to find the built-in potential barrier, \(V_{bi}\).

Figure 9.18 shows the ideal energy-band diagram for the nP abrupt heterojunction. The experimentally determined values of \(\Delta E_c\) and \(\Delta E_v\) may differ from the ideal values determined using the electron affinity rule. One possible explanation for this difference is that most heterojunctions have interface states. If we assume that the electrostatic potential is continuous through the junction, then the electric flux density will be discontinuous at the heterojunction due to the surface charge trapped in the interface states. The interface states will then change the energy-band diagram of the semiconductor heterojunction just as they changed the energy-band diagram of the metal–semiconductor junction. Another possible explanation for the deviation from the ideal is that as the two materials are brought together to form the heterojunction, the electron orbitals of each material begin to interact with each other, resulting in a transition region of a few angstroms at the interface. The energy bandgap is continuous through this transition region and not a characteristic of either material. However, we still have the relation

\[
\Delta E_c + \Delta E_v = \Delta E_g
\]

(9.50)

for the straddling type of heterojunction, although the \(\Delta E_c\) and \(\Delta E_v\) values may differ from those determined from the electron affinity rule.