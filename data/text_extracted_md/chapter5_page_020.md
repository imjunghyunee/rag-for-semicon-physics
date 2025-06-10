## Objective: Calculate the diffusion current density given a density gradient.

Assume that, in an n-type gallium arsenide semiconductor at \( T = 300 \, \text{K} \), the electron concentration varies linearly from \( 1 \times 10^{18} \) to \( 7 \times 10^{17} \, \text{cm}^{-3} \) over a distance of 0.10 cm. Calculate the diffusion current density if the electron diffusion coefficient is \( D_n = 225 \, \text{cm}^2/\text{s} \).

### Solution

The diffusion current density is given by

\[
J_{\text{n,diff}} = eD_n \frac{dn}{dx} = eD_n \frac{\Delta n}{\Delta x}
\]

\[
= (1.6 \times 10^{-19})(225) \left( \frac{1 \times 10^{18} - 7 \times 10^{17}}{0.10} \right) = 108 \, \text{A/cm}^2
\]

### Comment

A significant diffusion current density can be generated in a semiconductor material with only a modest density gradient.

### EXERCISE PROBLEM

**Ex 5.5** The hole density in silicon is given by \( p(x) = 10^{16} e^{-x/L_p} \, (x \geq 0) \) where \( L_p = 2 \times 10^{-4} \, \text{cm} \). Assume the hole diffusion coefficient is \( D_p = 8 \, \text{cm}^2/\text{s} \). Determine the hole diffusion current density at (a) \( x = 0 \), (b) \( x = 2 \times 10^{-4} \, \text{cm} \), and (c) \( x = 10^{-3} \, \text{cm} \).

----

### 5.2.2 Total Current Density

We now have four possible independent current mechanisms in a semiconductor. These components are electron drift and diffusion currents and hole drift and diffusion currents. The **total current density** is the sum of these four components, or, for the one-dimensional case,

\[
J = en_n \mu_n E_x + ep_n \mu_p E_x + eD_n \frac{dn}{dx} - eD_p \frac{dp}{dx} \tag{5.35}
\]

This equation may be generalized to three dimensions as

\[
J = en_n \mu_n \mathbf{E} + ep_n \mu_p \mathbf{E} + eD_n \nabla n - eD_p \nabla p \tag{5.36}
\]

The electron mobility gives an indication of how well an electron moves in a semiconductor as a result of the force of an electric field. The electron diffusion coefficient gives an indication of how well an electron moves in a semiconductor as a result of a density gradient. The electron mobility and diffusion coefficient are not independent parameters. Similarly, the hole mobility and diffusion coefficient are not independent parameters. The relationship between mobility and the diffusion coefficient is developed in the next section.

The expression for the total current in a semiconductor contains four terms. Fortunately in most situations, we will only need to consider one term at any one time at a particular point in a semiconductor.