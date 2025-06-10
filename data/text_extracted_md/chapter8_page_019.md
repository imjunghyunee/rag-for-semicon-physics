!Geometry of a "short" diode.

The steady-state excess minority carrier hole concentration in the n region is determined from Equation (8.9), which is given as

\[
\frac{d^2 (\delta p_n)}{dx^2} - \frac{\delta p_n}{L_p^2} = 0
\]

The original boundary condition at \( x = x_n \) still applies, given by Equation (8.11a) as

\[
p_n(x_n) = p_{n0} \exp \left( \frac{eV_a}{kT} \right)
\]

A second boundary condition needs to be determined. In many cases we assume that an ohmic contact exists at \( x = (x_n + W_n) \), implying an infinite surface-recombination velocity and therefore an excess minority carrier concentration of zero. The second boundary condition is then written as

\[
p_n(x = x_n + W_n) = p_{n0} \tag{8.30}
\]

The general solution to Equation (8.9) is again given by Equation (8.12), which was

\[
\delta p_n (x) = p_n(x) - p_{n0} = A e^{x/L_p} + B e^{-x/L_p} \quad (x \geq x_n)
\]

In this case, because of the finite length of the n region, both terms of the general solution must be retained. Applying the boundary conditions of Equations (8.11b) and (8.30), the excess minority carrier concentration is given by

\[
\delta p_n(x) = p_{n0} \left[ \exp \left( \frac{eV_a}{kT} \right) - 1 \right] \frac{\sinh \left[ (x_n + W_n - x)/L_p \right]}{\sinh[W_n/L_p]} \tag{8.31}
\]

Equation (8.31) is the general solution for the excess minority carrier hole concentration in the n region of a forward-biased pn junction. If \( W_n \gg L_p \), the assumption for the long diode, Equation (8.31) reduces to the previous result given by Equation (8.14). If \( W_n \ll L_p \), we can approximate the hyperbolic sine terms by

\[
\sinh \left[ \frac{x_n + W_n - x}{L_p} \right] \approx \frac{x_n + W_n - x}{L_p} \tag{8.32a}
\]

and

\[
\sinh \left[ \frac{W_n}{L_p} \right] \approx \frac{W_n}{L_p} \tag{8.32b}
\]

Then Equation (8.31) becomes

\[
\delta p_n(x) = p_{n0} \left[ \exp \left( \frac{eV_a}{kT} \right) - 1 \right] \frac{x_n + W_n - x}{W_n} \tag{8.33}
\]