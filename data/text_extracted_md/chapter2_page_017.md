## 2.3 Applications of Schrödinger's Wave Equation

For the incident wave, \( A_1 \cdot A_1^* \) is the probability density function of the incident particles. If we multiply this probability density function by the incident velocity, then \( v_1 \cdot A_1 \cdot A_1^* \) is the flux of incident particles in units of \#/cm\(^2\)-s. Likewise, the quantity \( v_1 \cdot B_1 \cdot B_1^* \) is the flux of the reflected particles, where \( v_1 \) is the velocity of the reflected wave. (The parameters \( v_1 \) and \( v_1 \) in these terms are actually the magnitudes of the velocity only.)

In region II, the potential is \( V = V_0 \). If we assume that \( E < V_0 \), then the differential equation describing the wave function in region II can be written as

\[
\frac{\partial^2 \psi_2(x)}{\partial x^2} - \frac{2m}{\hbar^2} (V_0 - E) \psi_2(x) = 0
\]

(2.43)

The general solution may then be written in the form

\[
\psi_2(x) = A_2 e^{-k_2 x} + B_2 e^{k_2 x} \quad (x \geq 0)
\]

(2.44)

where

\[
k_2 = \sqrt{\frac{2m(V_0 - E)}{\hbar^2}}
\]

(2.45)

One boundary condition is that the wave function \( \psi_2(x) \) must remain finite, which means that the coefficient \( B_2 = 0 \). The wave function is now given by

\[
\psi_2(x) = A_2 e^{-k_2 x} \quad (x \geq 0)
\]

(2.46)

The wave function at \( x = 0 \) must be continuous so that

\[
\psi_1(0) = \psi_2(0)
\]

(2.47)

Then from Equations (2.41), (2.46), and (2.47), we obtain

\[
A_1 + B_1 = A_2
\]

(2.48)

Since the potential function is everywhere finite, the first derivative of the wave function must also be continuous so that

\[
\left. \frac{\partial \psi_1}{\partial x} \right|_{x=0} = \left. \frac{\partial \psi_2}{\partial x} \right|_{x=0}
\]

(2.49)

Using Equations (2.41), (2.46), and (2.49), we obtain

\[
jk_1 A_1 - jk_1 B_1 = -k_2 A_2
\]

(2.50)

We can solve Equations (2.48) and (2.50) to determine the coefficients \( B_1 \) and \( A_2 \) in terms of the incident wave coefficient \( A_1 \). The results are

\[
B_1 = \left( \frac{-k_2^2 + 2jk_1 k_2 - k_1^2}{k_2^2 + k_1^2} \right) \cdot A_1
\]

(2.51a)

and

\[
A_2 = \frac{2k_1(k_1 - jk_2)}{k_2^2 + k_1^2} \cdot A_1
\]

(2.51b)