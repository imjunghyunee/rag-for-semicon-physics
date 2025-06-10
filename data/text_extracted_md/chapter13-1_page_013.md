to the I–V characteristics when the JFET is biased in the saturation region. This equation is used extensively in JFET applications and is given by

\[
I_D = I_{\text{DSS}} \left( 1 - \frac{V_{GS}}{V_p} \right)^2
\]

(13.14)

where \( I_{\text{DSS}} \) is the saturation current when \( V_{GS} = 0 \). At the end of this section, we compare the approximation given by Equation (13.14) and the ideal current–voltage equation that we have derived.

### I–V Derivation

The ideal current–voltage relationship of the JFET is derived by starting with Ohm’s law. Consider an n-channel JFET with the geometry shown in Figure 13.11. We are considering half of the two-sided symmetrical geometry. The differential resistance of the channel at a point \( x \) in the channel is

\[
dR = \frac{\rho dx}{A(x)}
\]

(13.15)

where \( \rho \) is the resistivity and \( A(x) \) is the cross-sectional area. If we neglect the minority carrier holes in the n channel, the channel resistivity is

\[
\rho = \frac{1}{e \mu_n N_d}
\]

(13.16)

The cross-sectional area is given by

\[
A(x) = [a - h(x)]W
\]

(13.17)

where \( W \) is the channel width. Equation (13.15) can now be written as

\[
dR = \frac{dx}{e \mu_n N_d [a - h(x)] W}
\]

(13.18)

The differential voltage across a differential length \( dx \) can be written as

\[
dV(x) = I_D dR(x)
\]

(13.19)

where the drain current \( I_D \) is constant through the channel. Substituting Equation (13.18) into Equation (13.19), we have

\[
dV(x) = \frac{I_D \, dx}{e \mu_n N_d W [a - h(x)]}
\]

(13.20a)

or

\[
I_D \, dx = e \mu_n N_d W [a - h(x)] \, dV(x)
\]

(13.20b)

The depletion width \( h(x) \) is given by

\[
h(x) = \left[ \frac{2 \varepsilon_s [V(x) + V_b - V_{GS}]}{e N_d} \right]^{1/2}
\]

(13.21)

where \( V(x) \) is the potential in the channel due to the drain-to-source voltage. Solving for \( V(x) \) in Equation (13.21) and taking the differential, we have

\[
dV(x) = \frac{e N_d h(x) \, dh(x)}{\varepsilon_s}
\]

(13.22)