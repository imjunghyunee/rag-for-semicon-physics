!Figure 8.21  
**Figure 8.21** Minority carrier concentration changes with changing forward-bias voltage.

The physics of the diffusion capacitance may be seen in Figure 8.21. The dc values of the minority carrier concentrations are shown along with the changes due to the ac component of voltage. The \(\Delta Q\) charge is alternately being charged and discharged through the junction as the voltage across the junction changes. The change in the stored minority carrier charge as a function of the change in voltage is the diffusion capacitance. One consequence of the approximations \(\omega \tau_p \ll 1\) and \(\omega \tau_n \ll 1\) is that there are no “wiggles” in the minority carrier curves. The sinusoidal frequency is low enough so that the exponential curves are maintained at all times.

----

**EXAMPLE 8.7**

**Objective:** Calculate the small-signal admittance parameters of a pn junction diode.

This example is intended to give an indication of the magnitude of the diffusion capacitance as compared with the junction capacitance considered in the last chapter. The diffusion resistance will also be calculated. Assume that \(N_D \gg N_A\) so that \(p_{n0} \gg n_{p0}\). This assumption implies that \(I_{p0} \gg I_{n0}\). Let \(T = 300 \, \text{K}\), \(\tau_{p0} = 10^{-7} \, \text{s}\), and \(I_{p0} = I_0 = 1 \, \text{mA}\).

**Solution**

The diffusion capacitance, with these assumptions, is given by

\[
C_d \approx \left( \frac{1}{2V_T} \right) (I_{p0} \tau_{p0}) = \frac{1}{(2)(0.0259)} (10^{-7} \times 10^{-3}) = 1.93 \times 10^{-9} \, \text{F}
\]

The diffusion resistance is

\[
r_d = \frac{V_T}{I_{p0}} = \frac{0.0259 \, \text{V}}{1 \, \text{mA}} = 25.9 \, \Omega
\]

**Comment**

The value of 1.93 nF for the diffusion capacitance of a forward-biased pn junction is three to four orders of magnitude larger than the junction capacitance of the reverse-biased pn junction, which we calculated in Example 7.5.