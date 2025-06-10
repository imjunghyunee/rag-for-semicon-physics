# 11.1 Nonideal Effects

In the ideal I–V relationship, current saturation occurs when the inversion charge density becomes zero at the drain terminal, or when

\[
V_{DS} = V_{DS(sat)} = V_{GS} - V_T
\]

(11.16)

for the n-channel MOSFET. However, velocity saturation can change this saturation condition. Velocity saturation will occur when the horizontal electric field is approximately \(10^4\) V/cm. If \(V_{DS} = 5\) V in a device with a channel length of \(L = 1 \, \mu m\), the average electric field is \(5 \times 10^4\) V/cm. Velocity saturation, then, is very likely to occur in short-channel devices.

The modified \(I_D(sat)\) characteristics are described approximately by

\[
I_D(sat) = W C_{ox} (V_{GS} - V_T) v_{sat}
\]

(11.17)

where \(v_{sat}\) is the saturation velocity (approximately \(10^7\) cm/s for electrons in bulk silicon) and \(C_{ox}\) is the gate oxide capacitance per cm². The saturation velocity will decrease somewhat with applied gate voltage because of the vertical electric field and surface scattering. Velocity saturation will yield an \(I_D(sat)\) value smaller than that predicted by the ideal relation, and it will yield a smaller \(V_{DS(sat)}\) value than predicted. The \(I_D(sat)\) current is also approximately linear with \(V_{GS}\), instead of having the ideal square law dependence predicted previously.

There are several models of mobility versus electric field. One particular relation that is commonly used is

\[
\mu = \frac{\mu_{eff}}{\left[1 + \left(\frac{\mu_{eff} E}{v_{sat}}\right)^n\right]^{1/n}}
\]

(11.18)

Figure 11.11 shows a comparison of drain current versus drain-to-source voltage characteristics for constant mobility and for field-dependent mobility. The smaller values of \(I_D(sat)\) and the approximate linear dependence on \(V_{GS}\) may be noted for the field-dependent mobility curves.

The transconductance is found from

\[
g_m = \frac{\partial I_D(sat)}{\partial V_{GS}} = W C_{ox} v_{sat}
\]

(11.19)

which is now independent of \(V_{GS}\) and \(V_{DS}\) when velocity saturation occurs. The drain current is saturated by the velocity saturation effect, which leads to a constant transconductance.

When velocity saturation occurs, the cutoff frequency is given by

\[
f_T = \frac{g_m}{2\pi C_G} = \frac{W C_{ox} v_{sat}}{2\pi (C_{ox} W L)} = \frac{v_{sat}}{2\pi L}
\]

(11.20)

where the parasitic capacitances are assumed to be negligible.

## 11.1.5 Ballistic Transport

Scattering events in a semiconductor limit the velocity of carriers to an average drift velocity as discussed in Chapter 5. The average drift velocity is a function of the