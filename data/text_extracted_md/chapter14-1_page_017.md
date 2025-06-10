# CHAPTER 14 Optical Devices

\(\delta n = \delta p = \delta p\). We will use \(\delta p\) as the concentration of excess carriers. In steady state, the excess carrier concentration is given by \(\delta p = G \tau_r\), where \(G\) is the generation rate of excess carriers (cm\(^{-3}\)s\(^{-1}\)) and \(\tau_r\) is the excess minority carrier lifetime.

The conductivity from Equation (14.18) can be rewritten as

\[
\sigma = e(\mu_n n_0 + \mu_p p_0) + e(\delta p)(\mu_n + \mu_p)
\]

(14.19)

The change in conductivity due to the optical excitation, known as the **photoconductivity**, is then

\[
\Delta \sigma = e(\delta p)(\mu_n + \mu_p)
\]

(14.20)

An electric field is induced in the semiconductor by the applied voltage, which produces a current. The current density can be written as

\[
J = (J_0 + J_L) = (\sigma_0 + \Delta \sigma)E
\]

(14.21)

where \(J_0\) is the current density in the semiconductor prior to optical excitation and \(J_L\) is the photocurrent density. The photocurrent density is \(J_L = \Delta \sigma \cdot E\). If the excess electrons and holes are generated uniformly throughout the semiconductor, then the photocurrent is given by

\[
I_L = J_L \cdot A = \Delta \sigma \cdot A E = eG\tau_r(\mu_n + \mu_p)AE
\]

(14.22)

where \(A\) is the cross-sectional area of the device. The photocurrent is directly proportional to the excess carrier generation rate, which in turn is proportional to the incident photon flux.

If excess electrons and holes are not generated uniformly throughout the semiconductor material, then the total photocurrent is found by integrating the photoconductivity over the cross-sectional area.

Since \(\mu_n E\) is the electron drift velocity, the electron transit time, that is, the time required for an electron to flow through the photoconductor, is

\[
t_n = \frac{L}{\mu_n E}
\]

(14.23)

The photocurrent, from Equation (14.22), can be rewritten as

\[
I_L = eG\tau_r \left( \frac{T_p}{t_n} \right) \left( 1 + \frac{\mu_p}{\mu_n} \right) AL
\]

(14.24)

We may define a photoconductor gain, \(\Gamma_p\), as the ratio of the rate at which charge is collected by the contacts to the rate at which charge is generated within the photoconductor. We can write the gain as

\[
\Gamma_p = \frac{I_L}{eG\tau_r AL}
\]

(14.25)

which, using Equation (14.24), can be written

\[
\Gamma_p = \frac{T_p}{t_n} \left( 1 + \frac{\mu_p}{\mu_n} \right)
\]

(14.26)