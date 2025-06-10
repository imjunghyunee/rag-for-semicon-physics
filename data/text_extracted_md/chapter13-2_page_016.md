# Chapter 13: The Junction Field-Effect Transistor

The density of the two-dimensional electron gas in a normal structure can be described using a charge control model. We may write:

\[
n_s = \frac{\varepsilon_l}{q(d + \Delta d)} (V_g - V_{\text{off}})
\]

(13.68)

where \(\varepsilon_l\) is the permittivity of the N-AlGaAs, \(d = d_t + d_i\) is the thickness of the doped-plus-undoped AlGaAs layer, and \(\Delta d\) is a correction factor given by:

\[
\Delta d = \frac{\varepsilon_a}{q} \approx 80 \, \text{Å}
\]

(13.69)

The threshold voltage \(V_{\text{off}}\) is given by:

\[
V_{\text{off}} = \phi_b - \frac{\Delta E_c}{q} - V_{pz}
\]

(13.70)

where \(\phi_b\) is the Schottky barrier height and \(V_{pz}\) is:

\[
V_{pz} = \frac{qN_ad_i^2}{2\varepsilon_l}
\]

(13.71)

A negative gate bias will reduce the two-dimensional electron gas concentration. If a positive gate voltage is applied, the density of the two-dimensional electron gas will increase. Increasing the gate voltage will increase the two-dimensional electron gas density until the conduction band of the AlGaAs crosses the Fermi level of the electron gas. Figure 13.29 shows this effect. At this point, the gate loses control over the electron gas since a parallel conduction path in the AlGaAs has been formed.

!Energy-band diagram of an enhancement mode HEMT

**Figure 13.29** | Energy-band diagram of an enhancement mode HEMT (a) with a slight forward gate voltage, and (b) with a larger forward gate voltage that creates a conduction channel in the AlGaAs. (From Fritzsche [5].)