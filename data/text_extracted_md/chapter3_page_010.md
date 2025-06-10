# 3.1 Allowed and Forbidden Energy Bands

Equation (3.19) relates the parameter \( k \) to the total energy \( E \) (through the parameter \( \alpha \)) and the potential function \( V_0 \) (through the parameter \( \beta \)).

As we mentioned, the more interesting solutions occur for \( E < V_0 \), which applies to the electron bound within the crystal. From Equation (3.7), the parameter \( \beta \) is then an imaginary quantity. We may define

\[
\beta = j\gamma
\]

where \( \gamma \) is a real quantity. Equation (3.19) can be written in terms of \( \gamma \) as

\[
\gamma^2 - \frac{\alpha^2}{\alpha\gamma} (\sin \alpha a)(\sinh \gamma b) + (\cos \alpha a)(\cosh \gamma b) = \cos k(a + b)
\]

Equation (3.21) does not lend itself to an analytical solution, but must be solved using numerical or graphical techniques to obtain the relation between \( k, E, \) and \( V_0 \).

The solution of Schrödinger’s wave equation for a single bound particle resulted in discrete allowed energies. The solution of Equation (3.21) will result in a band of allowed energies.

To obtain an equation that is more susceptible to a graphical solution and thus will illustrate the nature of the results, let the potential barrier width \( b \rightarrow 0 \) and the barrier height \( V_0 \rightarrow \infty \), but such that the product \( bV_0 \) remains finite. Equation (3.21) then reduces to

\[
\left( \frac{mV_0ba}{\hbar^2} \right) \frac{\sin \alpha a}{\alpha a} + \cos \alpha a = \cos ka
\]

We may define a parameter \( P' \) as

\[
P' = \frac{mV_0ba}{\hbar^2}
\]

Then, finally, we have the relation

\[
P' \frac{\sin \alpha a}{\alpha a} + \cos \alpha a = \cos ka
\]

Equation (3.24) again gives the relation between the parameter \( k \), total energy \( E \) (through the parameter \( \alpha \)), and the potential barrier \( bV_0 \). We may note that Equation (3.24) is not a solution of Schrödinger’s wave equation but gives the conditions for which Schrödinger’s wave equation will have a solution. If we assume that the crystal is infinitely large, then \( k \) in Equation (3.24) can assume a continuum of values and must be real.

## 3.1.3 The k-Space Diagram

To begin to understand the nature of the solution, initially consider the special case for which \( V_0 = 0 \). In this case \( P' = 0 \), which corresponds to a free particle since there are no potential barriers. From Equation (3.24), we have that

\[
\cos \alpha a = \cos ka
\]

or

\[
\alpha = k
\]