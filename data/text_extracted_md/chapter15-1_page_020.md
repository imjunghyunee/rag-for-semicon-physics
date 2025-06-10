## 15.5 Power MOSFETs

The current at the maximum power point (using the results from Example 15.1) is

\[
I_D = \frac{V_{DD}}{2R_D} = \frac{24}{2(4.8)} = 2.5 \, \text{A}
\]

and the corresponding drain-to-source voltage is

\[
V_{DS} = V_{DD} - I_D R_D = 24 - (2.5)(4.8) = 12 \, \text{V}
\]

The maximum power that may be dissipated in the transistor is \( P = V_{DS} I_D = (12)(2.5) = 30 \, \text{W} = P_T \), which corresponds to the maximum rated power. This point is shown on the curve.

The load line for the inverter circuit using device B is shown as curve B. The load line intersects the voltage axis at \( V_{DD} = 24 \, \text{V} \) as before. This curve can now intersect the current axis at the maximum rated drain current of \( I_{D, \text{max}} = 4 \, \text{A} \). We see that the load line falls within the SOA of the transistor.

For load line B, the drain resistance is

\[
R_D = \frac{V_{DD}}{I_D} = \frac{24}{4} = 6 \, \Omega
\]

The current at the maximum power point is

\[
I_D = \frac{V_{DD}}{2R_D} = \frac{24}{2(6)} = 2 \, \text{A}
\]

and the corresponding drain-to-source voltage is

\[
V_{DS} = V_{DD} - I_D R_D = 24 - (2)(6) = 12 \, \text{V}
\]

The maximum power that may be dissipated in the transistor is \( P = V_{DS} I_D = (12)(2) = 24 \, \text{W} \), which is less than the maximum rated power. This point is also shown on the curve.

### Conclusion

We see that if device A is used, the drain resistor is determined by the maximum power. However, if device B is used, the drain resistor is determined by the maximum rated current of the device.

### EXERCISE PROBLEM

**Ex 15.2** Consider the common-source circuit shown in Figure 15.26. Determine the required current, voltage, and power ratings of the MOSFET for (a) \( R_D = 12 \, \Omega \), \( V_{DD} = 24 \, \text{V} \) and (b) \( R_D = 8 \, \Omega \), \( V_{DD} = 40 \, \text{V} \).

\[
\forall \, \xi = -\frac{wq}{\lambda} \, \alpha \, \theta = \frac{5qAQ}{\alpha} \, \theta : \lambda \, \Xi = \lambda \, \theta \, \forall \, \xi = -\frac{wq}{\lambda} \, \alpha \, \theta \, \Xi = \frac{5qAQ}{\alpha} \, \theta \, \text{SVU}
\]

## 15.5.3 Parasitic BJT

The MOSFET has a parasitic BJT as an inherent part of its structure. The parasitic BJT may be seen in both the DMOS and VMOS structures shown in Figures 15.20 and 15.21. The source terminal corresponds to the n-type emitter, the p-type base or substrate region corresponds to the p-type base, and the n-type drain corresponds to the n-type collector. This is also shown schematically in Figure 15.28. The channel length of the MOSFET corresponds to the base width of the parasitic BJT. Since this length is normally quite small, the current gain \(\beta\) of the BJT can be larger than unity.