# CHAPTER 9 Metal–Semiconductor and Semiconductor Heterojunctions

## EXERCISE PROBLEM

**Ex 9.8** Repeat Example 9.8 for an n-Ge-to-P-GaAs heterojunction. The Ge is doped with \( N_d = 10^{15} \, \text{cm}^{-3} \) donors and the GaAs doped with \( N_a = 10^{15} \, \text{cm}^{-3} \) acceptors. Let \( T = 300 \, \text{K} \). (\( \lambda = 6890 = \lambda \, \text{SU}\))

We can determine the electric field and potential in the junction from Poisson’s equation in exactly the same way as we do for the homojunction. For homogeneous doping on each side of the junction, we have in the n region

\[
E_n = \frac{eN_d}{\epsilon_n} (x_n + x) \quad (-x_n \leq x < 0) \tag{9.41a}
\]

and in the P region

\[
E_p = \frac{eN_a}{\epsilon_p} (x_p - x) \quad (0 < x \leq x_p) \tag{9.41b}
\]

where \( \epsilon_n \) and \( \epsilon_p \) are the permittivities of the n and P materials, respectively. We may note that \( E_n = 0 \) at \( x = -x_n \), and \( E_p = 0 \) at \( x = x_p \). The electric flux density \( D \) is continuous across the junction, so

\[
\epsilon_n E_n(x = 0) = \epsilon_p E_p(x = 0) \tag{9.42a}
\]

which gives

\[
N_d x_n = N_a x_p \tag{9.42b}
\]

Equation (9.42b) simply states that the net negative charge in the P region is equal to the net positive charge in the n region—the same condition we had in a pn homojunction. We are neglecting any interface states that may exist at the heterojunction.

The electric potential can be found by integrating the electric field through the space charge region so that the potential difference across each region can be determined. We find that

\[
V_{bin} = \frac{eN_d x_n^2}{2\epsilon_n} \tag{9.43a}
\]

and

\[
V_{bip} = \frac{eN_a x_p^2}{2\epsilon_p} \tag{9.43b}
\]

Equation (9.42b) can be rewritten as

\[
\frac{x_n}{x_p} = \frac{N_a}{N_d} \tag{9.44}
\]

The ratio of the built-in potential barriers can then be determined as

\[
\frac{V_{bin}}{V_{bip}} = \frac{\epsilon_p}{\epsilon_n} \cdot \frac{N_a}{N_d} \cdot \frac{x_n^2}{x_p^2} = \frac{\epsilon_p N_a}{\epsilon_n N_d} \tag{9.45}
\]

Assuming that \( \epsilon_n \) and \( \epsilon_p \) are of the same order of magnitude, the larger potential difference is across the lower-doped region.