# 2.3 Applications of Schrödinger's Wave Equation

**Probability that the incident particle will penetrate the potential barrier and exist in region II. The probability of a particle penetrating the potential barrier is another difference between classical and quantum mechanics: The quantum mechanical penetration is classically not allowed.** Although there is a finite probability that the particle may penetrate the barrier, since the reflection coefficient in region I is unity, the particle in region II must eventually turn around and move back into region I.

## Objective

Calculate the penetration depth of a particle impinging on a potential barrier.

**Example 2.4**

Consider an incident electron that is traveling at a velocity of \(1 \times 10^6 \, \text{m/s}\) in region I.

### Solution

With \(V(x) = 0\), the total energy is also equal to the kinetic energy so that

\[
E = T = \frac{1}{2} mv^2 = 4.56 \times 10^{-21} \, \text{J} = 2.85 \times 10^{-2} \, \text{eV}
\]

Now, assume that the potential barrier at \(x = 0\) is twice as large as the total energy of the incident particle, or that \(V_0 = 2E\). The wave function solution in region II is \(\psi(x) = A_2 e^{-kx}\), where the constant \(k\) is given by \(k = \sqrt{2m(V_0 - E)}/\hbar\).

In this example, we want to determine the distance \(x = d\) at which the wave function magnitude has decayed to \(e^{-1}\) of its value at \(x = 0\). Then, for this case, we have \(kd = 1\) or

\[
1 = d \sqrt{\frac{2m(2E - E)}{\hbar^2}} = d \sqrt{\frac{2mE}{\hbar^2}}
\]

The distance is then given by

\[
d = \sqrt{\frac{\hbar^2}{2mE}} = \frac{1.054 \times 10^{-34}}{\sqrt{2(9.11 \times 10^{-31})(4.56 \times 10^{-21})}} = 11.6 \times 10^{-10} \, \text{m}
\]

or

\[
d = 11.6 \, \text{Å}
\]

### Comment

This penetration distance corresponds to approximately two lattice constants of silicon. The numbers used in this example are rather arbitrary. We used a distance at which the wave function decayed to \(e^{-1}\) of its initial value. We could have arbitrarily used \(e^{-2}\), for example, but the results give an indication of the magnitude of penetration depth.

### Exercise Problem

**Ex 2.4** The probability of finding a particle at a distance \(d\) in region II compared with that at \(x = 0\) is given by \(\exp(-2kd)\). Consider an electron traveling in region I at a velocity of \(10^6 \, \text{m/s}\) incident on a potential barrier whose height is three times the kinetic energy of the electron. Find the probability of finding the electron at a distance \(d\) compared with \(x = 0\) where \(d\) is (a) 10 Å and (b) 100 Å into the potential barrier.

The case when the total energy of a particle, which is incident on the potential barrier, is greater than the barrier height, or \(E > V_0\), is left as an exercise at the end of the chapter.