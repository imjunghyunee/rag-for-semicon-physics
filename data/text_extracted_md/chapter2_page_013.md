## 2.3 Applications of Schrödinger's Wave Equation

!Figure 2.6  
**Figure 2.6** Potential function of the infinite potential well.

A particular form of solution to this equation is given by

\[
\psi(x) = A_1 \cos kx + A_2 \sin kx
\]

(2.29)

where

\[
k = \sqrt{\frac{2mE}{\hbar^2}}
\]

(2.30)

One boundary condition is that the wave function \(\psi(x)\) must be continuous so that

\[
\psi(x = 0) = \psi(x = a) = 0
\]

(2.31)

Applying the boundary condition at \(x = 0\), we must have that \(A_1 = 0\). At \(x = a\), we have

\[
\psi(x = a) = 0 = A_2 \sin ka
\]

(2.32)

This equation is valid if \(ka = n\pi\), where the parameter \(n\) is a positive integer, or \(n = 1, 2, 3, \ldots\). The parameter \(n\) is referred to as a quantum number. We can write

\[
k = \frac{n\pi}{a}
\]

(2.33)

Negative values of \(n\) simply introduce a negative sign in the wave function and yield redundant solutions for the probability density function. We cannot physically distinguish any difference between \(+n\) and \(-n\) solutions. Because of this redundancy, negative values of \(n\) are not considered.

The coefficient \(A_2\) can be found from the normalization boundary condition that was given by Equation (2.18) as \(\int_0^a \psi^*(x) \psi(x) \, dx = 1\). If we assume that the wave