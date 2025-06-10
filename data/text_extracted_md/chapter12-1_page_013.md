# 12.2 Minority Carrier Distribution

!Figure 12.14  
**Figure 12.14** Minority carrier distribution in an npn bipolar transistor operating in the forward-active mode.

In Figure 12.14, as there are two n regions, we have minority carrier holes in both emitter and collector. To distinguish between these two minority carrier hole distributions, we use the notation shown in the figure. Keep in mind that we are dealing only with minority carriers. The parameters \( p_{E0}, n_{B0}, \) and \( p_{C0} \) denote the thermal-equilibrium minority carrier concentrations in the emitter, base, and collector, respectively. The functions \( p_E(x'), n_B(x), \) and \( p_C(x'') \) denote the steady-state minority carrier concentrations in the emitter, base, and collector, respectively. We assume that the neutral collector length \( x_C \) is long compared to the minority carrier diffusion length \( L_C \) in the collector, but we take into account a finite emitter length \( x_E \). If we assume that the surface recombination velocity at \( x' = x_E \) is infinite, then the excess minority carrier concentration at \( x' = x_E \) is zero, or \( p_E(x' = x_E) = p_{E0} \). An infinite surface recombination velocity is a good approximation when an ohmic contact is fabricated at \( x' = x_E \).

## Base Region

The steady-state excess minority carrier electron concentration is found from the ambipolar transport equation, which we discussed in detail in Chapter 6. For a zero electric field in the neutral base region, the ambipolar transport equation in steady state reduces to

\[
D_B \frac{\partial^2 (\delta n_B(x))}{\partial x^2} - \frac{\delta n_B(x)}{\tau_{B0}} = 0
\]

(12.9)

where \( \delta n_B \) is the excess minority carrier electron concentration, and \( D_B \) and \( \tau_{B0} \) are the minority carrier diffusion coefficient and lifetime in the base region, respectively. The excess electron concentration is defined as

\[
\delta n_B(x) = n_B(x) - n_{B0}
\]

(12.10)

The general solution to Equation (12.9) can be written as

\[
\delta n_B(x) = A \exp \left( \frac{x}{L_B} \right) + B \exp \left( \frac{-x}{L_B} \right)
\]

(12.11)