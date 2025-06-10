# 3.1 Allowed and Forbidden Energy Bands

The parameter \( k \) is called a constant of motion and will be considered in more detail as we develop the theory. The function \( u(x) \) is a periodic function with period \( (a + b) \).

We stated in Chapter 2 that the total solution to the wave equation is the product of the time-independent solution and the time-dependent solution, or

\[
\Psi(x, t) = \psi(x)\phi(t) = u(x)e^{ikx} \cdot e^{-iEt/\hbar}
\]

(3.2)

which may be written as

\[
\Psi(x, t) = u(x)e^{i(kx-Et/\hbar)}
\]

(3.3)

This traveling-wave solution represents the motion of an electron in a single-crystal material. The amplitude of the traveling wave is a periodic function and the parameter \( k \) is also referred to as a wave number.

We can now begin to determine a relation between the parameter \( k \), the total energy \( E \), and the potential \( V_0 \). If we consider region I in Figure 3.6 (0 < x < a) in which \( V(x) = 0 \), take the second derivative of Equation (3.1), and substitute this result into the time-independent Schrödinger’s wave equation given by Equation (2.13), we obtain the relation

\[
\frac{d^2u(x)}{dx^2} + 2ijk \frac{du(x)}{dx} - (k^2 - \alpha^2)u(x) = 0
\]

(3.4)

The function \( u_1(x) \) is the amplitude of the wave function in region I and the parameter \( \alpha \) is defined as

\[
\alpha^2 = \frac{2mE}{\hbar^2}
\]

(3.5)

Consider now a specific region II, \(-b < x < 0\), in which \( V(x) = V_0 \), and apply Schrödinger’s wave equation. We obtain the relation

\[
\frac{d^2u(x)}{dx^2} + 2ijk \frac{du(x)}{dx} - \left[k^2 - \frac{2mV_0}{\hbar^2}\right]u(x) = 0
\]

(3.6)

where \( u_2(x) \) is the amplitude of the wave function in region II. We may define

\[
\frac{2m}{\hbar^2}(E - V_0) = \alpha^2 - \frac{2mV_0}{\hbar^2} = \beta^2
\]

(3.7)

so that Equation (3.6) may be written as

\[
\frac{d^2u(x)}{dx^2} + 2ijk \frac{du(x)}{dx} - (k^2 - \beta^2)u(x) = 0
\]

(3.8)

Note that from Equation (3.7), if \( E \geq V_0 \), the parameter \( \beta \) is real, whereas if \( E < V_0 \), then \( \beta \) is imaginary.

The solution to Equation (3.4), for region I, is of the form

\[
u_1(x) = Ae^{\alpha x} + Be^{-\alpha x} \quad \text{for} \quad (0 < x < a)
\]

(3.9)

and the solution to Equation (3.8), for region II, is of the form

\[
u_2(x) = Ce^{\beta x} + De^{-\beta x} \quad \text{for} \quad (-b < x < 0)
\]

(3.10)