## 2.3 Applications of Schrödinger's Wave Equation

We will utilize the resulting concepts later in the discussion of semiconductor properties.

### 2.3.1 Electron in Free Space

As a first example of applying the Schrödinger's wave equation, consider the motion of an electron in free space. If there is no force acting on the particle, then the potential function \( V(x) \) will be constant and we must have \( E > V(x) \). Assume, for simplicity, that the potential function \( V(x) = 0 \) for all \( x \). Then, the time-independent wave equation can be written from Equation (2.13) as

\[
\frac{\partial^2 \psi(x)}{\partial x^2} + \frac{2mE}{\hbar^2} \psi(x) = 0
\]

(2.19)

The solution to this differential equation can be written in the form

\[
\psi(x) = A \exp \left( j \frac{\sqrt{2mE}}{\hbar} x \right) + B \exp \left( -j \frac{\sqrt{2mE}}{\hbar} x \right)
\]

(2.20a)

or

\[
\psi(x) = A \exp(jkx) + B \exp(-jkx)
\]

(2.20b)

where

\[
k = \frac{\sqrt{2mE}}{\hbar}
\]

(2.21)

and is called a wave number.

Recall that the time-dependent portion of the solution is

\[
\phi(t) = e^{-j\omega t} = e^{-j\omega t}
\]

(2.22)

Then the total solution for the wave function is given by

\[
\Psi(x, t) = A \exp[j(kx - \omega t)] + B \exp[-j(kx + \omega t)]
\]

(2.23)

This wave function solution is a traveling wave, which means that a particle moving in free space is represented by a traveling wave. The first term, with the coefficient \( A \), is a wave traveling in the \( +x \) direction, while the second term, with the coefficient \( B \), is a wave traveling in the \( -x \) direction. The value of these coefficients will be determined from boundary conditions. We will again see the traveling-wave solution for an electron in a crystal or semiconductor material.

Assume, for a moment, that we have a particle traveling in the \( +x \) direction, which will be described by the \( +x \) traveling wave. The coefficient \( B = 0 \). We can write the traveling-wave solution in the form

\[
\Psi(x, t) = A \exp[j(kx - \omega t)]
\]

(2.24)

where \( k \) is the wave number given by

\[
k = \frac{\sqrt{2mE}}{\hbar} = \frac{\sqrt{p^2}}{\hbar} = \frac{p}{\hbar}
\]

(2.25a)