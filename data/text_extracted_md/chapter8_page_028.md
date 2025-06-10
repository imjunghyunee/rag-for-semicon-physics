# 8.2 Generation–Recombination Currents and High-Injection Levels

!Graph

**Figure 8.16** Ideal diffusion, recombination, and total current in a forward-biased pn junction.

However, as the forward-bias voltage increases, the excess carrier concentrations increase and may become comparable or even greater than the majority carrier concentration. From Equation (8.18), we can write

\[
np = n_i^2 \exp\left(\frac{V_a}{V_t}\right)
\]

We have that \( n = n_0 + \delta n \) and \( p = p_0 + \delta p \), so that

\[
(n_0 + \delta n)(p_0 + \delta p) = n_i^2 \exp\left(\frac{V_a}{V_t}\right) \tag{8.60}
\]

Under high-level injection, we may have \( \delta n > n_0 \) and \( \delta p > p_0 \) so that Equation (8.60) becomes approximately

\[
(\delta n)(\delta p) \approx n_i^2 \exp\left(\frac{V_a}{V_t}\right) \tag{8.61}
\]

Since \( \delta n = \delta p \), then

\[
\delta n = \delta p \approx n_i \exp\left(\frac{V_a}{2V_t}\right) \tag{8.62}
\]

The diode current is proportional to the excess carrier concentration so that, under high-level injection, we have

\[
I \propto \exp\left(\frac{V_a}{2V_t}\right) \tag{8.63}
\]