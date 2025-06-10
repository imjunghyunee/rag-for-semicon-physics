# Chapter 8 The pn Junction Diode

!Figure 8.20

**Figure 8.20** The dc characteristics of a forward-biased pn junction used in the small-signal admittance calculations.

where \( V_0 \) is the dc quiescent bias voltage and \( v_l(t) \) is the ac signal voltage that is superimposed on this dc level. We may now write

\[
p_n(x = 0) = p_{n0} \exp \left[ \frac{e(V_0 + v_l(t))}{kT} \right] = p_n(0, t)
\]

(8.70)

Equation (8.70) may be written as

\[
p_n(0, t) = p_{n0} \exp \left[ \frac{ev_l(t)}{kT} \right]
\]

(8.71)

where

\[
p_{n0} = p_{n0} \exp \left( \frac{eV_0}{kT} \right)
\]

(8.72)

If we assume that \( |v_l(t)| \ll (kT/e) = V_T \), then the exponential term in Equation (8.71) may be expanded into a Taylor series retaining only the linear terms, and the minority carrier hole concentration at \( x = 0 \) can be written as

\[
p_n(0, t) \approx p_{n0} \left[ 1 + \frac{v_l(t)}{V_T} \right]
\]

(8.73)

If we assume that the time-varying voltage \( v_l(t) \) is a sinusoidal signal, we can write Equation (8.73) as

\[
p_n(0, t) = p_{n0} \left[ 1 + \frac{\hat{V}_l}{V_T} e^{j\omega t} \right]
\]

(8.74)

where \( \hat{V}_l \) is the phasor of the applied sinusoidal voltage. Equation (8.74) will be used as the boundary condition in the solution of the time-dependent diffusion equation for the minority carrier holes in the n region.

In the neutral n region (\( x > 0 \)), the electric field is assumed to be zero; thus, the behavior of the excess minority carrier holes is determined from the equation

\[
D_p \frac{\partial^2 (\delta p_n)}{\partial x^2} - \frac{\delta p_n}{\tau_{p0}} = \frac{\partial (\delta p_n)}{\partial t}
\]

(8.75)