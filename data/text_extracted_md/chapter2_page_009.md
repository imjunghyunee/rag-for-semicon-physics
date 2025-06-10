## 2.2 Schrödinger's Wave Equation

ask what the relation is between the function and the electron. The total wave function is the product of the position-dependent, or time-independent, function and the time-dependent function. From Equation (2.7) we have

\[
\Psi(x, t) = \psi(x)\phi(t) = \psi(x)e^{-iEt/\hbar} = \psi(x)e^{-i\omega t}
\]

(2.14)

Since the total wave function \(\Psi(x, t)\) is a complex function, it cannot by itself represent a real physical quantity.

Max Born postulated in 1926 that the function \(|\Psi(x, t)|^2 \, dx\) is the probability of finding the particle between \(x\) and \(x + dx\) at a given time, or that \(|\Psi(x, t)|^2\) is a probability density function. We have that

\[
|\Psi(x, t)|^2 = \Psi(x, t) \cdot \Psi^*(x, t)
\]

(2.15)

where \(\Psi^*(x, t)\) is the complex conjugate function. Therefore

\[
\Psi^*(x, t) = \psi^*(x) \cdot e^{iEt/\hbar}
\]

Then the product of the total wave function and its complex conjugate is given by

\[
\Psi(x, t)\Psi^*(x, t) = [\psi(x)e^{-iEt/\hbar}] [\psi^*(x)e^{iEt/\hbar}] = \psi(x)\psi^*(x)
\]

(2.16)

Therefore, we have that

\[
|\Psi(x, t)|^2 = \psi(x)\psi^*(x) = |\psi(x)|^2
\]

(2.17)

is the probability density function and is independent of time. One major difference between classical and quantum mechanics is that in classical mechanics, the position of a particle or body can be determined precisely, whereas in quantum mechanics, the position of a particle is found in terms of a probability. We will determine the probability density function for several examples, and since this property is independent of time, we will, in general, only be concerned with the time-independent wave function.

### 2.2.3 Boundary Conditions

Since the function \(|\psi(x)|^2\) represents the probability density function, then for a single particle, we must have

\[
\int_{-\infty}^{\infty} |\psi(x)|^2 \, dx = 1
\]

(2.18)

The probability of finding the particle somewhere is certain. Equation (2.18) allows us to normalize the wave function and is one boundary condition that is used to determine some wave function coefficients.

The remaining boundary conditions imposed on the wave function and its derivative are postulates. However, we may state the boundary conditions and present arguments that justify why they must be imposed. The wave function and its first derivative must have the following properties if the total energy \(E\) and the potential \(V(x)\) are finite everywhere.

- **Condition 1.** \(\psi(x)\) must be finite, single-valued, and continuous.
- **Condition 2.** \(\frac{\partial \psi(x)}{\partial x}\) must be finite, single-valued, and continuous.