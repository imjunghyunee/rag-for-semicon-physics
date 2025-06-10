### 8.1 pn Junction Current

#### Exercise Problem

**Ex 8.1**  
A silicon pn junction at \( T = 300 \, \text{K} \) is doped with impurity concentrations of \( N_d = 2 \times 10^{16} \, \text{cm}^{-3} \) and \( N_a = 5 \times 10^{16} \, \text{cm}^{-3} \). The junction is forward biased at \( V_a = 0.650 \, \text{V} \). Determine the minority carrier concentrations at the space charge edges. Does low injection still apply?  
\[ \text{[S^a \cdot t \cdot uw} \, 0.1 \times ZG \, 8 = (\gamma \cdot d \cdot uw \, 0.1) \times LS \cdot \varepsilon = (t-x-y \, t+su \, v] \]

The minority carrier concentrations at the space charge edges, given by Equations (8.6) and (8.7), were derived assuming that a forward-bias voltage (\( V_a > 0 \)) was applied across the pn junction. However, nothing in the derivation prevents \( V_a \) from being negative (reverse bias). If a reverse-biased voltage greater than a few tenths of a volt is applied to the pn junction, then we see from Equations (8.6) and (8.7) that the minority carrier concentrations at the space charge edge are essentially zero. The minority carrier concentrations for the reverse-biased condition drop below the thermal-equilibrium values.

### 8.1.4 Minority Carrier Distribution

We developed, in Chapter 6, the ambipolar transport equation for excess minority carrier holes in an n region. This equation, in one dimension, is

\[
D_p \frac{\partial^2 (\delta p_n)}{\partial x^2} - \mu_p E \frac{\partial (\delta p_n)}{\partial x} + g' - \frac{\delta p_n}{\tau_{p0}} = \frac{\partial (\delta p_n)}{\partial t}
\]

(8.8)

where \( \delta p_n = p_n - p_{n0} \) is the excess minority carrier hole concentration and is the difference between the total and thermal equilibrium minority carrier concentrations. The ambipolar transport equation describes the behavior of excess carriers as a function of time and spatial coordinates.

In Chapter 5, we calculated drift current densities in a semiconductor. We determined that relatively large currents could be created with fairly small electric fields. As a first approximation, we assume that the electric field is zero in both the neutral p and n regions. In the n region for \( x > x_n \), we have that \( E = 0 \) and \( g' = 0 \). If we also assume steady state so \( \partial (\delta p_n)/\partial t = 0 \), then Equation (8.8) reduces to

\[
\frac{d^2 (\delta p_n)}{dx^2} - \frac{\delta p_n}{L_{p}^2} = 0 \quad (x > x_n)
\]

(8.9)

where \( L_{p}^2 = D_p \tau_{p0} \). For the same set of conditions, the excess minority carrier electron concentration in the p region is determined from

\[
\frac{d^2 (\delta n_p)}{dx^2} - \frac{\delta n_p}{L_{n}^2} = 0 \quad (x < x_p)
\]

(8.10)

where \( L_{n}^2 = D_n \tau_{n0} \).