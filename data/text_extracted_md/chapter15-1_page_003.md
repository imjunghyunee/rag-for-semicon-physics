# Chapter 15: Semiconductor Microwave and Power Devices

!Figure 15.2  
**Figure 15.2** | Equivalent circuit of the tunnel diode.

Figure 15.1b shows an expanded plot of the I–V characteristics in the tunneling range. A point is shown on the curve where the minimum value of negative resistance occurs. (Note that \( R_{\text{min}} \) is a positive quantity.) The equivalent circuit of the tunnel diode for the case when the diode is biased at the \(-R_{\text{min}}\) point is shown in Figure 15.2. The parameter \( C_j \) is the junction capacitance, and the parameters \( L_p \) and \( R_p \) are the parasitic or interconnect line inductance and resistance, respectively.

The small signal input impedance can be written as

\[
Z = \left[ R_p - \frac{R_{\text{min}}}{1 + \omega^2 R_{\text{min}}^2 C_j^2} \right] + j\omega \left[ L_p - \frac{\omega R_{\text{min}}^2 C_j}{1 + \omega^2 R_{\text{min}}^2 C_j^2} \right]
\]

(15.1)

The resistive part of the impedance goes to zero at a frequency of

\[
f_r = \frac{1}{2\pi R_{\text{min}} C_j} \sqrt{\frac{R_{\text{min}}}{R_p} - 1}
\]

(15.2)

For frequencies \( f > f_r \), the resistive part of the impedance becomes positive so that the diode loses its negative differential resistance characteristic. The operating frequency must then occur at \( f_0 < f_r \). The frequency \( f_r \) is referred to as the maximum resistive cutoff frequency.

The tunneling process is a majority carrier effect so the diode does not exhibit time delays due to minority carrier diffusion, which means that the diode is capable of operating at microwave frequencies. However, due to the relatively small voltage range in which the diode exhibits the negative resistance characteristic, the tunnel diode is not used extensively.

## 15.2 Gunn Diode

Another negative differential resistance device is the GUNN diode, or Transferred-Electron Device (TED). The transferred-electron phenomenon is demonstrated in a few semiconductors in which conduction electrons in a high-mobility band are scattered to a low-mobility band by a high electric field. In Chapter 5, we discussed the drift velocity of electrons in GaAs versus electric field. Figure 15.3 again shows a plot of this characteristic. InP also shows this same characteristic.

Figure 15.4 shows an expanded plot of the energy-band structure in GaAs that is given in Figure 5.8. For small electric fields, essentially all of the electrons in the