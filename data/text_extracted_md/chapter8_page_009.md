# Chapter 8: The pn Junction Diode

The boundary conditions for the total minority carrier concentrations are

\[
p_n(x_n) = p_{n0} \exp\left(\frac{eV_a}{kT}\right) \tag{8.11a}
\]

\[
n_p(-x_p) = n_{p0} \exp\left(\frac{eV_a}{kT}\right) \tag{8.11b}
\]

\[
p_n(x \to +\infty) = p_{n0} \tag{8.11c}
\]

\[
n_p(x \to -\infty) = n_{p0} \tag{8.11d}
\]

As minority carriers diffuse from the space charge edge into the neutral semiconductor regions, they recombine with majority carriers. We assume that the lengths \(W_n\) and \(W_p\) shown in Figure 8.3a are very long, meaning in particular that \(W_n \gg L_n\) and \(W_p \gg L_p\). The excess minority carrier concentrations must approach zero at distances far from the space charge region. The structure is referred to as a long pn junction.

The general solution to Equation (8.9) is

\[
\delta p_n(x) = p_n(x) - p_{n0} = A e^{+x/L_p} + B e^{-x/L_p} \quad (x \geq x_n) \tag{8.12}
\]

and the general solution to Equation (8.10) is

\[
\delta n_p(x) = n_p(x) - n_{p0} = C e^{+x/L_n} + D e^{-x/L_n} \quad (x \leq -x_p) \tag{8.13}
\]

Applying the boundary conditions from Equations (8.11c) and (8.11d), the coefficients \(A\) and \(D\) must be zero. The coefficients \(B\) and \(C\) may be determined from the boundary conditions given by Equations (8.11a) and (8.11b). The excess carrier concentrations are then found to be, for \(x \geq x_n\),

\[
\delta p_n(x) = p_n(x) - p_{n0} = p_{n0} \left[\exp\left(\frac{eV_a}{kT}\right) - 1\right] \exp\left(\frac{x - x_n}{L_p}\right) \tag{8.14}
\]

and, for \(x \leq -x_p\),

\[
\delta n_p(x) = n_p(x) - n_{p0} = n_{p0} \left[\exp\left(\frac{eV_a}{kT}\right) - 1\right] \exp\left(\frac{x + x_p}{L_n}\right) \tag{8.15}
\]

The minority carrier concentrations decay exponentially with distance away from the junction to their thermal-equilibrium values. Figure 8.5 shows these results. Again, we have assumed that both the n-region and the p-region lengths are long compared to the minority carrier diffusion lengths.

In Chapter 6, we discussed the concept of quasi-Fermi levels, which apply to excess carriers in a nonequilibrium condition. Since excess electrons exist in the neutral p region and excess holes exist in the neutral n region, we can apply quasi-Fermi levels to these regions. We had defined quasi-Fermi levels in terms of carrier concentrations as

\[
p = p_0 + \delta p = n_i \exp\left(\frac{E_{Fi} - E_{Fp}}{kT}\right) \tag{8.16}
\]