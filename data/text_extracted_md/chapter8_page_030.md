# 8.3.1 Diffusion Resistance

The ideal current–voltage relationship of the pn junction diode was given by Equation (8.27), where \( J \) and \( J_s \) are current densities. If we multiply both sides of the equation by the junction cross-sectional area, we have

\[
I_D = I_s \left[ \exp \left( \frac{eV_a}{kT} \right) - 1 \right]
\]

(8.64)

where \( I_D \) is the diode current and \( I_s \) is the diode reverse-saturation current.

Assume that the diode is forward-biased with a dc voltage \( V_0 \) producing a dc diode current \( I_{DQ} \). If we now superimpose a small, low-frequency sinusoidal voltage as shown in Figure 8.18, then a small sinusoidal current will be produced, superimposed on the dc current. The ratio of sinusoidal current to sinusoidal voltage is called the incremental conductance. In the limit of a very small sinusoidal current and voltage, the small-signal incremental conductance is just the slope of the dc current–voltage curve, or

\[
g_d = \left. \frac{dI_D}{dV_a} \right|_{I_D=I_{DQ}}
\]

(8.65)

The reciprocal of the incremental conductance is the incremental resistance, defined as

\[
r_d = \left. \frac{dV_a}{dI_D} \right|_{I_D=I_{DQ}}
\]

(8.66)

where \( I_{DQ} \) is the dc quiescent diode current.

!Figure 8.18

**Figure 8.18** Curve showing the concept of the small-signal diffusion resistance.