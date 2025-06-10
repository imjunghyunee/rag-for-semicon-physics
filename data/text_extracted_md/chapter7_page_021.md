# 7.4 Junction Breakdown

!Figure 7.14

**Figure 7.14** Critical electric field at breakdown in a one-sided junction as a function of impurity doping concentrations. (From Sze and Ng [14].)

The depletion width \( x_n \) is given approximately as

\[
x_n \approx \left[ \frac{2 \varepsilon_s V_r}{e \cdot N_d} \right]^{1/2}
\]

(7.60)

where \( V_r \) is the magnitude of the applied reverse-biased voltage. We have neglected the built-in potential \( V_{bi} \).

If we now define \( V_B \) to be the breakdown voltage, \( V_B \), the maximum electric field, \( E_{\text{max}} \), will be defined as a critical electric field, \( E_{\text{crit}} \), at breakdown. Combining Equations (7.59) and (7.60), we may write

\[
V_B = \frac{\varepsilon_s E_{\text{crit}}}{2eN_B}
\]

(7.61)

where \( N_B \) is the semiconductor doping in the low-doped region of the one-sided junction. The critical electric field, plotted in Figure 7.14, is a slight function of doping.

We have been considering a uniformly doped planar junction. The breakdown voltage will decrease for a linearly graded junction. (See Section 7.5.) Figure 7.15 shows a plot of the breakdown voltage for a one-sided abrupt junction and a linearly graded junction. If we take into account the curvature of a diffused junction as well, the breakdown voltage will be further degraded.

----

**Objective:** Design an ideal one-sided n\(^+\)p junction diode to meet a breakdown voltage specification.

Consider a silicon pn junction diode at \( T = 300 \, K \). Assume that \( N_d = 3 \times 10^{18} \, \text{cm}^{-3} \). Design the diode such that the breakdown voltage is \( V_B = 100 \, V \).

**Solution**

From Figure 7.15, we find that the doping concentration in the low-doped side of a one-sided abrupt junction should be approximately \( 4 \times 10^{15} \, \text{cm}^{-3} \) for a breakdown voltage of 100 V.

----

**DESIGN EXAMPLE 7.7**