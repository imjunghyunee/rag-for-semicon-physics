## CHAPTER 6 Non-equilibrium Excess Carriers in Semiconductors

Equation (6.110) may be used as a boundary condition to the general solution given by Equation (6.109) in Example 6.8. Using Figure 6.18, we have that at \( \bar{t} = -1 \), and Equation (6.110) becomes

\[
D_p \frac{d(\delta p)}{dx} \bigg|_{\text{surf}} = s \delta p|_{\text{surf}}
\]

(6.111)

We have argued that the coefficient \( A \) is zero in Equation (6.109). Then, from Equation (6.109), we can write that

\[
\delta p_{\text{surf}} = \delta p(0) = g' \tau_{p0} + B
\]

(6.112a)

and

\[
\frac{d(\delta p)}{dx} \bigg|_{\text{surf}} = \frac{d(\delta p)}{dx} \bigg|_{x=0} = -\frac{B}{L_p}
\]

(6.112b)

Substituting Equations (6.112a) and (6.112b) into Equation (6.111) and solving for the coefficient \( B \), we obtain

\[
B = -\frac{s g' \tau_{p0}}{(D_p/L_p) + s}
\]

(6.113)

The excess minority carrier hole concentration can then be written as

\[
\delta p(x) = g' \tau_{p0} \left[ 1 - \frac{s L_p e^{-x/L_p}}{D_p + s L_p} \right]
\]

(6.114)

----

### EXAMPLE 6.10

**Objective:** Determine the value of surface recombination velocity corresponding to the parameters given in Example 6.9.

From Example 6.9, we have that \( g' \tau_{p0} = 10^{14} \, \text{cm}^{-3} \), \( D_p = 10 \, \text{cm}^2/\text{s} \), \( L_p = 31.6 \, \mu\text{m} \), and \( \delta p(0) = 10^{13} \, \text{cm}^{-3} \).

**Solution**

Writing Equation (6.114) at the surface, we have

\[
\delta p(0) = g' \tau_{p0} \left[ 1 - \frac{s}{(D_p/L_p) + s} \right]
\]

Solving for the surface recombination velocity, we find that

\[
s = \frac{D_p}{L_p} \left( \frac{g' \tau_{p0}}{\delta p(0)} - 1 \right)
\]

which becomes

\[
s = \frac{10}{31.6 \times 10^{-4}} \left[ \frac{10^{14}}{10^{13}} - 1 \right] = 2.85 \times 10^4 \, \text{cm/s}
\]