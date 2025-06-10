# Chapter 2: Introduction to Quantum Mechanics

where \( \psi(x) \) is a function of the position \( x \) only and \( \Phi(t) \) is a function of time \( t \) only. Substituting this form of the solution into Schrödinger’s wave equation, we obtain

\[
-\frac{\hbar^2}{2m} \Phi(x) \frac{\partial^2 \psi(x)}{\partial x^2} + V(x) \psi(x) \Phi(t) = j\hbar \psi(x) \frac{\partial \Phi(t)}{\partial t}
\]

(2.8)

If we **divide by the total wave function**, Equation (2.8) becomes

\[
-\frac{\hbar^2}{2m} \frac{1}{\psi(x)} \frac{\partial^2 \psi(x)}{\partial x^2} + V(x) = j\hbar \cdot \frac{1}{\Phi(t)} \cdot \frac{\partial \Phi(t)}{\partial t}
\]

(2.9)

Since the left side of Equation (2.9) is a function of position \( x \) only and the right side of the equation is a function of time \( t \) only, each side of this equation must be equal to a constant. We will denote this separation of variables constant by \( \eta \).

The time-dependent portion of Equation (2.9) is then written as

\[
\eta = j\hbar \cdot \frac{1}{\Phi(t)} \cdot \frac{\partial \Phi(t)}{\partial t}
\]

(2.10)

where again the parameter \( \eta \) is called a separation constant. **The solution of Equation (2.10) can be written in the form**

\[
\Phi(t) = e^{-j\eta t/\hbar}
\]

(2.11a)

The form of this solution is the classical exponential form of a sinusoidal wave. We have that \( E = h\nu \) or \( E = h\omega/2\pi \). Then \( \omega = \eta/\hbar = E/\hbar \) so that the separation constant is equal to the total energy \( E \) of the particle.

We can then write

\[
\Phi(t) = e^{-jEt/\hbar} = e^{-j\omega t}
\]

(2.11b)

We see that \( \omega = E/\hbar \) and is the radian or angular frequency of the sinusoidal wave.

The time-independent portion of Schrödinger’s wave equation can now be written from Equation (2.9) as

\[
-\frac{\hbar^2}{2m} \cdot \frac{1}{\psi(x)} \cdot \frac{\partial^2 \psi(x)}{\partial x^2} + V(x) = E
\]

(2.12)

where the separation constant is the total energy \( E \) of the particle. Equation (2.12) may be written as

\[
\frac{\partial^2 \psi(x)}{\partial x^2} + \frac{2m}{\hbar^2} (E - V(x)) \psi(x) = 0
\]

(2.13)

where again \( m \) is the mass of the particle, \( V(x) \) is the potential experienced by the particle, and \( E \) is the total energy of the particle. **This time-independent Schrödinger’s wave equation can also be justified on the basis of the classical wave equation as shown in Appendix E.** The pseudo-derivation in the appendix is a simple approach but shows the plausibility of the time-independent Schrödinger’s equation.

## 2.2.2 Physical Meaning of the Wave Function

We are ultimately trying to use the wave function \( \Psi(x, t) \) to describe the behavior of an electron in a crystal. The function \( \Psi(x, t) \) is a wave function, so it is reasonable to...