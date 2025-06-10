# Introduction to Quantum Mechanics

The reflected probability density function is given by

\[
B_i \cdot B_i^* = \frac{(k_2^2 - k_1^2 + 2ijk_1k_2)(k_2^2 - k_1^2 - 2jk_1k_2)}{(k_2^2 + k_1^2)^2} \cdot A_i \cdot A_i^*
\]

(2.52)

We can define a reflection coefficient, \( R \), as the ratio of the reflected flux to the incident flux, which is written as

\[
R = \frac{v_r \cdot B_i \cdot B_i^*}{v_i \cdot A_i \cdot A_i^*}
\]

(2.53)

where \( v_i \) and \( v_r \) are the incident and reflected velocities, respectively, of the particles. In region I, \( V = 0 \) so that \( E = T \), where \( T \) is the kinetic energy of the particle. The kinetic energy is given by

\[
T = \frac{1}{2} mv^2
\]

(2.54)

so that the constant \( k_i \), from Equation (2.42) may be written as

\[
k_i = \sqrt{\frac{2m}{\hbar^2} \left(\frac{1}{2} mv^2\right)} = \sqrt{m^2 \frac{v^2}{\hbar^2}} = \frac{mv}{\hbar}
\]

(2.55)

The incident velocity can then be written as

\[
v_i = \frac{\hbar}{m} \cdot k_i
\]

(2.56)

Since the reflected particle also exists in region I, the reflected velocity (magnitude) is given by

\[
v_r = \frac{\hbar}{m} \cdot k_i
\]

(2.57)

The incident and reflected velocities (magnitudes) are equal. The reflection coefficient is then

\[
R = \frac{v_r \cdot B_i \cdot B_i^*}{v_i \cdot A_i \cdot A_i^*} = \frac{B_i \cdot B_i^*}{A_i \cdot A_i^*}
\]

(2.58)

Substituting the expression from Equation (2.52) into Equation (2.58), we obtain

\[
R = \frac{B_i \cdot B_i^*}{A_i \cdot A_i^*} = \frac{k_2^2 - k_1^2}{k_2^2 + k_1^2} + \frac{4k_1^2k_2^2}{k_2^2 + k_1^2} = 1.0
\]

(2.59)

The result of \( R = 1 \) implies that all of the particles incident on the potential barrier for \( E < V_0 \) are eventually reflected. Particles are not absorbed or transmitted through the potential barrier. This result is entirely consistent with classical physics and one might ask why we should consider this problem in terms of quantum mechanics. The interesting result is in terms of what happens in region II.

The wave solution in region II was given by Equation (2.46) as \( \psi(x) = A_2 e^{-\kappa x} \). The coefficient \( A_2 \) from Equation (2.48) is \( A_2 = A_i + B_i \), which we derived from the boundary conditions. For the case of \( E < V_0 \), the coefficient \( A_2 \) is not zero. If \( A_2 \) is not zero, then the probability density function \( \psi_2(x) \cdot \psi_2^*(x) \) of the particle being found in region II is not equal to zero. This result implies that there is a finite probability of finding the particle in region II.