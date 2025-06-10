# Chapter 13: The Junction Field-Effect Transistor

The built-in potential barrier is

\[
V_{bi} = V_t \ln \left( \frac{N_a N_d}{n_i^2} \right) = (0.0259) \ln \left( \frac{(10^{16})(10^5)}{(1.5 \times 10^{10})^2} \right) = 0.814 \, \text{V}
\]

The pinch-off voltage, from Equation (13.4), is then found as

\[
V_p = V_{bi} - V_{po} = 0.814 - 4.35 = -3.54 \, \text{V}
\]

**Comment**

The pinch-off voltage, or gate-to-source voltage to achieve pinch-off, for the n-channel depletion mode device is a negative quantity as we have said.

## Exercise Problem

**Ex 13.1** A silicon n-channel JFET at \( T = 300 \, \text{K} \) has a gate doping concentration of \( N_a = 10^{18} \, \text{cm}^{-3} \) and a channel doping concentration of \( N_d = 2 \times 10^{16} \, \text{cm}^{-3} \). Determine the metallurgical channel thickness, \( a \), such that the pinch-off voltage is \( V_p = -2.50 \, \text{V} \).

The pinch-off voltage is the gate-to-source voltage that must be applied to turn the JFET off and must be within the voltage range of the circuit design. The magnitude of the pinch-off voltage must also be less than the breakdown voltage of the junction.

## p-channel pn JFET

Figure 13.10b shows a p-channel JFET with the same basic geometry as the n-channel JFET we considered. The induced depletion region for the one-sided n\(^+\)p junction is again denoted by \( h \) and is given by

\[
h = \left[ \frac{2 \varepsilon_s (V_{bi} + V_{GS})}{e N_a} \right]^{1/2} \tag{13.5}
\]

For a reverse-biased n\(^+\)p junction, \( V_{GS} \) must be positive. The internal pinch-off voltage is again defined to be the total pn junction voltage to achieve pinch-off, so that when \( h = a \) we have

\[
a = \left[ \frac{2 \varepsilon_s V_{po}}{e N_a} \right]^{1/2} \tag{13.6}
\]

or

\[
V_{po} = \frac{ea^2 N_a}{2 \varepsilon_s} \tag{13.7}
\]

The internal pinch-off voltage for the p-channel device is also defined to be a positive quantity.

The pinch-off voltage is again defined as the gate-to-source voltage to achieve the pinch-off condition. For the p-channel depletion mode device, we have, from Equation (13.5), at pinch-off

\[
V_{dA} + V_p = V_{po} \quad \text{or} \quad V_p = V_{po} - V_{dA} \tag{13.8}
\]

The pinch-off voltage for a p-channel depletion mode JFET is a positive quantity.