# 13.2 The Device Characteristics

Equation (13.27) can also be written as

\[
I_{D} = G_{o1} \left\{ V_{DS} - \frac{2}{3} \sqrt{\frac{1}{V_{po}}} \left[ (V_{DS} + V_{bi} - V_{GS})^2 - (V_{bi} - V_{GS})^2 \right] \right\}
\]

(13.30)

where

\[
G_{o1} = \frac{\mu_n (eN_d)^{3/2} W_a^3}{2e_s L V_{po}} = \frac{e \mu_n N_d W_a}{L} = \frac{3I_{p1}}{V_{po}}
\]

(13.31)

The channel conductance is defined as

\[
g_d = \frac{\partial I_D}{\partial V_{DS}} \bigg|_{V_{DS}=0}
\]

(13.32)

Taking the derivative of Equation (13.30) with respect to \(V_{DS}\), we obtain

\[
g_d = \frac{\partial I_D}{\partial V_{DS}} \bigg|_{V_{DS}=0} = G_{o1} \left[ 1 - \left( \frac{V_{bi} - V_{GS}}{V_{po}} \right)^2 \right]
\]

(13.33)

We may note from Equation (13.33) that \(G_{o1}\) would be the conductance of the channel if both \(V_{bi}\) and \(V_{GS}\) were zero. This condition would exist if no space charge regions existed in the channel. We may also note, from Equation (13.33), that the channel conductance is modulated or controlled by the gate voltage. This channel conductance modulation is the basis of the field-effect phenomenon.

We have shown that the drain becomes pinched off, for the n-channel JFET, when

\[
V_{DS} = V_{DS(sat)} = V_{po} - (V_{bi} - V_{GS})
\]

(13.34)

In the saturation region, the saturation drain current is determined by setting \(V_{DS} = V_{DS(sat)}\) in Equation (13.29) so that

\[
I_{D} = I_{D(sat)} = I_{p1} \left[ 1 - \frac{3}{2} \left( \frac{V_{bi} - V_{GS}}{V_{po}} \right) \right] \left[ 1 - \frac{2}{3} \sqrt{\frac{V_{bi} - V_{GS}}{V_{po}}} \right]
\]

(13.35)

The ideal saturation drain current is independent of the drain-to-source voltage. Figure 13.12 shows the ideal current–voltage characteristics of a silicon n-channel JFET.

----

**Objective:** Calculate the maximum current in an n-channel JFET.

**Example 13.3**

Consider a silicon n-channel JFET at \(T = 300 \, K\) with the following parameters: \(N_d = 10^{18} \, \text{cm}^{-3}\), \(N_a = 10^{16} \, \text{cm}^{-3}\), \(a = 0.75 \, \mu m\), \(L = 10 \, \mu m\), \(W = 30 \, \mu m\), and \(\mu_n = 1000 \, \text{cm}^2/\text{V}\cdot\text{s}\).

**Solution**

The pinch-off current from Equation (13.28) becomes

\[
I_{p1} = \frac{(1000)((1.6 \times 10^{-19})(10^{18}))(30 \times 10^{-4})(0.75 \times 10^{-4})}{6(11.7)(8.85 \times 10^{-14})(10^{16} \times 10^{-6})} = 0.522 \, \text{mA}
\]

We also have from Example 13.1 that \(V_{bi} = 0.814 \, V\) and \(V_{po} = 4.35 \, V\). The maximum current occurs when \(V_{GS} = 0\), so from Equation (13.35)

\[
I_{D(max)} = I_{p1} \left[ 1 - \frac{3}{2} \left( \frac{V_{bi}}{V_{po}} \right) \right] \left[ 1 - \frac{2}{3} \sqrt{\frac{V_{bi}}{V_{po}}} \right]
\]

(13.36)