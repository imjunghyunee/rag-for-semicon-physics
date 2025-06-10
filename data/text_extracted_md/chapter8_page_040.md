### 8.4 Charge Storage and Diode Transients

!Figure 8.24  
**Figure 8.24** Simple circuit for switching a diode from forward to reverse bias.

#### 8.4.1 The Turn-off Transient

Suppose we want to switch a diode from the forward bias on state to the reverse-biased off state. Figure 8.24 shows a simple circuit that will switch the applied bias at \( t = 0 \). For \( t < 0 \), the forward-bias current is

\[
I = I_F = \frac{V_F - V_a}{R_F}
\]

(8.107)

The minority carrier concentrations in the device, for the applied forward voltage \( V_F \), are shown in Figure 8.25a. There is excess minority carrier charge stored in both the p and n regions of the diode. The excess minority carrier concentrations at the space charge edges are supported by the forward-bias junction voltage \( V_a \). When the voltage is switched from the forward- to the reverse-biased state, the excess minority carrier concentrations at the space charge edges can no longer be supported and they start to decrease, as shown in Figure 8.25b.

The collapse of the minority carrier concentrations at the edges of the space charge region leads to large concentration gradients and diffusion currents in the reverse-biased direction. If we assume, for the moment, that the voltage across the diode junction is small compared with \( V_R \), then the reverse-biased current is limited to approximately

\[
I = -I_R \approx -\frac{V_R}{R_R}
\]

(8.108)

The junction capacitances do not allow the junction voltage to change instantaneously. If the current \( I_R \) were larger than this value, there would be a forward-bias voltage across the junction, which would violate our assumption of a reverse-biased current.