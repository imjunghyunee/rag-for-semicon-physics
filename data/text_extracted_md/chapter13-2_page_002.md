# CHAPTER 13 The Junction Field-Effect Transistor

!Figure 13.14

**Figure 13.14** | Experimental and theoretical \(\sqrt{I_D}\) versus \(V_{GS}\) characteristics of an enhancement mode JFET.

The square root of Equation (13.47), or \(\sqrt{I_{D(sat)}}\) versus \(V_{GS}\), is plotted as the ideal dotted curve shown in Figure 13.14. The ideal curve intersects the voltage axis at the threshold voltage, \(V_r\). The solid line shows an experimental plot. Equation (13.46) does not describe the experimental results well near the threshold voltage. The ideal current–voltage relationship is derived assuming an abrupt depletion approximation for the pn junction. However, when the depletion region extends almost through the channel, a more accurate model of the space charge region must be used to more accurately predict the drain current characteristics near threshold. We consider the subthreshold conduction in Section 13.3.3.

## DESIGN EXAMPLE 13.7

**Objective:** Design the channel width of an n-channel GaAs enhancement-mode pn JFET to produce a specified current for a given bias.

Consider the GaAs JFET described in Example 13.6. In addition, assume \(\mu_e = 8000 \, \text{cm}^2/\text{V·s}\) and \(L = 1.2 \, \mu\text{m}\). Design the width such that \(I_{D} = 75 \, \mu\text{A}\) with an applied voltage of \(V_{GS} = 0.5 \, \text{V}\).

### Solution

In the saturation region, the current is given by

\[
I_{D} = k_s(V_{GS} - V_r)^2
\]

or

\[
75 \times 10^{-6} = k_s(0.5 - 0.24)^2
\]

The conduction parameter is then

\[
k_s = 1.109 \, \text{mA/V}^2
\]

The conduction parameter, from Equation (13.48), is given by

\[
k_s = \frac{\mu_e \varepsilon W}{2aL}
\]