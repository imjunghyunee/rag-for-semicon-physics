## 7.3 Reverse Applied Bias

In the reverse-biased voltage \(dV_R\) will uncover additional positive charges in the n region and additional negative charges in the p region. The junction capacitance is defined as

\[
C' = \frac{dQ'}{dV_R}
\]

(7.38)

where

\[
dQ' = eN_d \, dx_n = eN_a \, dx_p
\]

(7.39)

The differential charge \(dQ'\) is in units of C/cm\(^2\) so that the capacitance \(C'\) is in units of farads per square centimeter F/cm\(^2\), or capacitance per unit area.

For the total potential barrier, Equation (7.28) may be written as

\[
x_n = \left\{ \frac{2 \varepsilon_s (V_{bi} + V_R)}{e} \left[ \frac{N_a}{N_d} \right] \left[ \frac{1}{N_a + N_d} \right] \right\}^{1/2}
\]

(7.40)

The junction capacitance can be written as

\[
C' = \frac{dQ'}{dV_R} = eN_d \frac{dx_n}{dV_R}
\]

(7.41)

so that

\[
C' = \left\{ \frac{e \varepsilon_s N_d N_a}{2(V_{bi} + V_R)(N_a + N_d)} \right\}^{1/2}
\]

(7.42)

Exactly the same capacitance expression is obtained by considering the space charge region extending into the p region \(x_p\). The junction capacitance is also referred to as the **depletion layer capacitance**.

----

### Objective

Calculate the junction capacitance of a pn junction.

### Example 7.5

Consider the same pn junction as that in Example 7.3. Again assume that \(V_R = 5 \, \text{V}\).

#### Solution

The junction capacitance is found from Equation (7.42) as

\[
C' = \left\{ \frac{(1.6 \times 10^{-19})(11.7)(8.85 \times 10^{-14})(10^{16})(10^{15})}{2(0.635 + 5)(10^{16} + 10^{15})} \right\}^{1/2}
\]

or

\[
C' = 3.66 \times 10^{-7} \, \text{F/cm}^2
\]

If the cross-sectional area of the pn junction is, for example, \(A = 10^{-4} \, \text{cm}^2\), then the total junction capacitance is

\[
C = C' \cdot A = 0.366 \times 10^{-12} \, \text{F} = 0.366 \, \text{pF}
\]

#### Comment

The value of junction capacitance is usually in the pF, or smaller, range.