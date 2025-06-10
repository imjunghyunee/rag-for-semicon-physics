# Exercise Problem

**Ex 7.5** Consider a GaAs pn junction at \( T = 300 \, \text{K} \) doped to \( N_d = 5 \times 10^{15} \, \text{cm}^{-3} \) and \( N_a = 2 \times 10^{16} \, \text{cm}^{-3} \).  
(a) Calculate \( V_{bi} \).  
(b) Determine the junction capacitance \( C' \) for \( V_R = 4 \, \text{V} \).  
(c) Repeat part (b) for \( V_R = 8 \, \text{V} \).

\[
\left[ \frac{\varepsilon_s}{W} \right] \cdot 6.01 \times 9.9 = \lambda \left( \frac{\varepsilon_s}{W} \right) \cdot 6.01 \times 8.7 = \lambda \left( \frac{\varepsilon_s}{W} \right) \cdot 9.1 = \lambda \left( \frac{\varepsilon_s}{W} \right) \cdot \text{suV}
\]

If we compare Equation (7.34) for the total depletion width \( W \) of the space charge region under reverse bias and Equation (7.42) for the junction capacitance \( C' \), we find that we can write

\[
C' = \frac{\varepsilon_s}{W}
\]

(7.43)

Equation (7.43) is the same as the capacitance per unit area of a parallel plate capacitor. Considering Figure 7.9, we may have come to this same conclusion earlier. Keep in mind that the space charge width is a function of the reverse-biased voltage so that the junction capacitance is also a function of the reverse-biased voltage applied to the pn junction.

## 7.3.3 One-Sided Junctions

Consider a special pn junction called the one-sided junction. If, for example, \( N_a \gg N_d \), this junction is referred to as a p\(^+\)n junction. The total space charge width, from Equation (7.34), reduces to

\[
W \approx \left[ \frac{2 \varepsilon_s (V_R + V_{bi})}{e N_d} \right]^{1/2}
\]

(7.44)

Considering the expressions for \( x_n \) and \( x_p \), we have for the p\(^+\)n junction

\[
x_p \ll x_n
\]

(7.45)

and

\[
W \approx x_n
\]

(7.46)

Almost the entire space charge layer extends into the low-doped region of the junction. This effect can be seen in Figure 7.10.

The junction capacitance of the p\(^+\)n junction reduces to

\[
C' \approx \left[ \frac{e \varepsilon_s N_d}{2(V_R + V_{bi})} \right]^{1/2}
\]

(7.47)

The depletion layer capacitance of a one-sided junction is a function of the doping concentration in the low-doped region. Equation (7.47) may be manipulated to give

\[
\left( \frac{1}{C'} \right)^2 = \frac{2(V_R + V_{bi})}{e \varepsilon_s N_d}
\]

(7.48)

which shows that the inverse capacitance squared is a linear function of applied reverse-biased voltage.