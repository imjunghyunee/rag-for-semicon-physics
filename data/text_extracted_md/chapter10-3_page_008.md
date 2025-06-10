# Fundamentals of the Metal–Oxide–Semiconductor Field-Effect Transistor

Equation (10.70) can also be written as

\[
I_D = \frac{k'_p}{2} \cdot \frac{W}{L} \cdot \left[ 2(V_{SG} + V_T)V_{SD} - V_{SD}^2 \right] = K_p \left[ 2(V_{SG} + V_T)V_{SD} - V_{SD}^2 \right] \tag{10.71}
\]

where \( K_p = \mu_p C_{ox} \) is the process conduction parameter for the p-channel MOSFET and \( K_p = (W \mu_p C_{ox})/(2L) = (k'_p/2) \cdot (W/L) \) is the conduction parameter.

When the transistor is biased in the saturation region, the I–V relation is given by

\[
I_D(sat) = \frac{W \mu_p C_{ox}}{2L} (V_{SG} + V_T)^2 \tag{10.72}
\]

Equation (10.72) is valid for \( V_{SD} \geq V_{SD} (sat) \).

Equation (10.72) can also be written as

\[
I_D = \frac{k'_p}{2} \cdot \frac{W}{L} \cdot (V_{SG} + V_T)^2 = K_p (V_{SG} + V_T)^2 \tag{10.73}
\]

The source-to-drain saturation voltage is given by

\[
V_{SD}(sat) = V_{SG} + V_T \tag{10.74}
\]

Note the change in the sign in front of \( V_T \) and note that the mobility is now the mobility of the holes in the hole inversion layer charge. Keep in mind that \( V_T \) is negative for a p-channel enhancement mode MOSFET and positive for a depletion mode p-channel device.

One assumption we made in the derivation of the current–voltage relationship was that the charge neutrality condition given by Equation (10.50) was valid over the entire length of the channel. We implicitly assumed that \( Q_{ox}(max) \) was constant along the length of the channel. The space charge width, however, varies between source and drain due to the drain-to-source voltage; it is widest at the drain when \( V_{DS} > 0 \). A change in the space charge density along the channel length must be balanced by a corresponding change in the inversion layer charge. An increase in the space charge width means that the inversion layer charge is reduced, implying that the drain current and drain-to-source saturation voltage are less than the ideal values. The actual saturation drain current may be as much as 20 percent less than the predicted value due to this bulk charge effect.

## 10.3.4 Transconductance

The MOSFET transconductance is defined as the change in drain current with respect to the corresponding change in gate voltage, or

\[
g_m = \frac{\partial I_D}{\partial V_{GS}} \tag{10.75}
\]

The transconductance is sometimes referred to as the transistor gain.