# CHAPTER 7 The pn Junction

where \( M_n \) is a multiplication factor. The hole current is increasing through the depletion region from the n to p region and reaches a maximum value at \( x = 0 \). The total current is constant through the pn junction in steady state.

We can write an expression for the incremental electron current at some point \( x \) as

\[
dI_n(x) = I_n(x) \alpha_n dx + I_p(x) \alpha_p dx \tag{7.50}
\]

where \( \alpha_n \) and \( \alpha_p \) are the electron and hole ionization rates, respectively. The ionization rates are the number of electron-hole pairs generated per unit length by an electron (\( \alpha_n \)) or by a hole (\( \alpha_p \)). Equation (7.50) may be written as

\[
\frac{dI_n(x)}{dx} = I_n(x) \alpha_n + I_p(x) \alpha_p \tag{7.51}
\]

The total current \( I \) is given by

\[
I = I_n(x) + I_p(x) \tag{7.52}
\]

which is a constant. Solving for \( I_p(x) \) from Equation (7.52) and substituting into Equation (7.51), we obtain

\[
\frac{dI_n(x)}{dx} + (\alpha_p - \alpha_n) I_n(x) = \alpha_p I \tag{7.53}
\]

If we make the assumption that the electron and hole ionization rates are equal so that

\[
\alpha_n = \alpha_p = \alpha \tag{7.54}
\]

then Equation (7.53) may be simplified and integrated through the space charge region. We will obtain

\[
I_n(W) - I_n(0) = I \int_0^W \alpha \, dx \tag{7.55}
\]

Using Equation (7.49), Equation (7.55) may be written as

\[
\frac{M_n I_{n0} - I_n(0)}{I} = \int_0^W \alpha \, dx \tag{7.56}
\]

Since \( M_n I_{n0} \approx I \) and since \( I_n(0) = I_{n0} \), Equation (7.56) becomes

\[
1 - \frac{1}{M_n} = \int_0^W \alpha \, dx \tag{7.57}
\]

The avalanche breakdown voltage is defined to be the voltage at which \( M_n \) approaches infinity. The avalanche breakdown condition is then given by

\[
\int_0^W \alpha \, dx = 1 \tag{7.58}
\]

The ionization rates are strong functions of electric field and, since the electric field is not constant through the space charge region, Equation (7.58) is not easy to evaluate.

If we consider, for example, a one-sided p\(^+\)n junction, the maximum electric field is given by

\[
E_{\text{max}} = \frac{eN_d x_n}{\varepsilon_s} \tag{7.59}
\]