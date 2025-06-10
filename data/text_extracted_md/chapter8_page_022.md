### 8.2 Generation–Recombination Currents and High-Injection Levels

are generated, they are swept out of the space charge region by the electric field. The flow of charge is in the direction of a reverse-biased current. This **reverse-biased generation current**, caused by the generation of electrons and holes in the space charge region, is in addition to the ideal reverse-biased saturation current.

We may calculate the density of the reverse-biased generation current by considering Equation (8.36). If we make a simplifying assumption and let the trap level be at the intrinsic Fermi level, then from Equations (6.92) and (6.97), we have that \( n' = n_i \) and \( p' = n_i \). Equation (8.36) now becomes

\[
R = \frac{-n_i}{1 + \frac{1}{N_C \tau_p} + \frac{1}{N_V \tau_n}}
\]

(8.37)

Using the definitions of lifetimes from Equations (6.103) and (6.104), we may write Equation (8.37) as

\[
R = \frac{-n_i}{\tau_{p0} + \tau_{n0}}
\]

(8.38)

If we define a new lifetime as the average of \( \tau_{p0} \) and \( \tau_{n0} \), or

\[
\tau_0 = \frac{\tau_{p0} + \tau_{n0}}{2}
\]

(8.39)

then the recombination rate can be written as

\[
R = \frac{-n_i}{2\tau_0} = -G
\]

(8.40)

The negative recombination rate implies a generation rate, so \( G \) is the generation rate of electrons and holes in the space charge region.

The generation current density may be determined from

\[
J_{gen} = \int_0^W eG \, dx
\]

(8.41)

where the integral is over the space charge region. If we assume that the generation rate is constant throughout the space charge region, then we obtain

\[
J_{gen} = \frac{en_i W}{2\tau_0}
\]

(8.42)

The total reverse-biased current density is the sum of the ideal reverse saturation current density and the generation current density, or

\[
J_r = J_s + J_{gen}
\]

(8.43)

The ideal reverse-saturation current density \( J_s \) is independent of the reverse-biased voltage. However, \( J_{gen} \) is a function of the depletion width \( W \), which in turn is a function of the reverse-biased voltage. The actual reverse-biased current density, then, is no longer independent of the reverse-biased voltage.