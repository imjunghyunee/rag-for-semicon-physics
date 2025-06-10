# Chapter 8: The pn Junction Diode

If we divide the equation by \( V_t = kT/e \), take the exponential of both sides, and then take the reciprocal, we obtain

\[
\frac{n_i^2}{N_D N_A} = \exp\left(-\frac{eV_0}{kT}\right) \tag{8.1}
\]

If we assume complete ionization, we can write

\[
n_{n0} \approx N_D \tag{8.2}
\]

where \( n_{n0} \) is the thermal-equilibrium concentration of majority carrier electrons in the n region. In the p region, we can write

\[
n_{p0} \approx \frac{n_i^2}{N_A} \tag{8.3}
\]

where \( n_{p0} \) is the thermal-equilibrium concentration of minority carrier electrons. Substituting Equations (8.2) and (8.3) into Equation (8.1), we obtain

\[
n_{p0} = n_{n0} \exp\left(-\frac{eV_0}{kT}\right) \tag{8.4}
\]

This equation relates the minority carrier electron concentration on the p side of the junction to the majority carrier electron concentration on the n side of the junction in thermal equilibrium.

If a positive voltage is applied to the p region with respect to the n region, the potential barrier is reduced. Figure 8.3a shows a pn junction with an applied voltage \( V_a \). The electric field in the bulk p and n regions is normally very small. Essentially all of the applied voltage is across the junction region. The electric field \( E_{\text{app}} \) induced by the applied voltage is in the opposite direction to the thermal-equilibrium space charge electric field, so the net electric field in the space charge region is reduced below the equilibrium value. The delicate balance between diffusion and the E-field

!Figure 8.3

**Figure 8.3** (a) A pn junction with an applied forward-bias voltage showing the directions of the electric field induced by \( V_a \) and the space charge electric field. (b) Energy-band diagram of the forward-biased pn junction.