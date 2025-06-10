# CHAPTER 8 The pn Junction Diode

The fact that we now have drift current densities in the p and n regions implies that the electric field in these regions is not zero as we had originally assumed. We can calculate the electric field in the neutral regions and determine the validity of our zero-field approximation.

## EXAMPLE 8.4

**Objective:** Calculate the electric field in a neutral region of a silicon diode to produce a given majority carrier drift current density.

Consider a silicon pn junction at \( T = 300 \, \text{K} \) with the parameters given in Example 8.2 and with an applied forward-bias voltage \( V_s = 0.65 \, \text{V} \).

### Solution

The total forward-bias current density is given by

\[
J = J_s \left[ \exp \left( \frac{eV}{kT} \right) - 1 \right]
\]

We determined the reverse-saturation current density in Example 8.2, so we can write

\[
J = (4.155 \times 10^{-11}) \left[ \exp \left( \frac{0.65}{0.0259} \right) - 1 \right] = 3.295 \, \text{A/cm}^2
\]

The total current far from the junction in the n region will be majority carrier electron drift current, so we can write

\[
J = J_n \approx e \mu_n N_d E
\]

The doping concentration is \( N_d = 10^{16} \, \text{cm}^{-3} \), and, if we assume \( \mu_n = 1350 \, \text{cm}^2/\text{V-s} \), then the electric field must be

\[
E = \frac{J_n}{e \mu_n N_d} = \frac{3.295}{(1.6 \times 10^{-19})(1350)(10^{16})} = 1.525 \, \text{V/cm}
\]

### Comment

We assumed, in the derivation of the current–voltage equation, that the electric field in the neutral p and n regions was zero. Although the electric field is not zero, this example shows that the magnitude is very small—thus the approximation of zero electric field is very good.

### EXERCISE PROBLEM

Ex 8.4 Determine the electric field in the neutral n region and neutral p region for the GaAs pn junction diode described in Ex 8.3.

## 8.1.7 Temperature Effects

The ideal reverse-saturation current density \( J_s \), given by Equation (8.26), is a function of the thermal-equilibrium minority carrier concentrations \( n_{p0} \) and \( p_{n0} \). These minority carrier concentrations are proportional to \( n_i^2 \), which is a very strong function of temperature. For a silicon pn junction, the ideal reverse-saturation current density will increase by approximately a factor of 4 for every \( 10^\circ \text{C} \) increase in temperature.