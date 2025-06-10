## 2.4 Extensions of the Wave Theory to Atoms

The potential function is due to the coulomb attraction between the proton and electron and is given by

\[
V(r) = -\frac{e^2}{4\pi \varepsilon_0 r}
\]

(2.64)

where \( e \) is the magnitude of the electronic charge and \( \varepsilon_0 \) is the permittivity of free space. This potential function, although spherically symmetric, leads to a three-dimensional problem in spherical coordinates.

We may generalize the time-independent Schrödinger’s wave equation to three dimensions by writing

\[
\nabla^2 \psi(r, \theta, \phi) + \frac{2m_0}{\hbar^2} (E - V(r)) \psi(r, \theta, \phi) = 0
\]

(2.65)

where \( \nabla^2 \) is the Laplacian operator and must be written in spherical coordinates for this case. The parameter \( m_0 \) is the rest mass of the electron. In spherical coordinates, Schrödinger’s wave equation may be written as

\[
\frac{1}{r^2} \frac{\partial}{\partial r} \left( r^2 \frac{\partial \psi}{\partial r} \right) + \frac{1}{r^2 \sin \theta} \frac{\partial}{\partial \theta} \left( \sin \theta \frac{\partial \psi}{\partial \theta} \right) + \frac{1}{r^2 \sin^2 \theta} \frac{\partial^2 \psi}{\partial \phi^2} + \frac{2m_0}{\hbar^2} (E - V(r)) \psi = 0
\]

(2.66)

The solution to Equation (2.66) can be determined by the separation-of-variables technique. We will assume that the solution to the time-independent wave equation can be written in the form

\[
\psi(r, \theta, \phi) = R(r) \cdot \Theta(\theta) \cdot \Phi(\phi)
\]

(2.67)

where \( R \), \( \Theta \), and \( \Phi \) are functions only of \( r \), \( \theta \), and \( \phi \), respectively. Substituting this form of solution into Equation (2.66), we will obtain

\[
\sin^2 \theta \left[ \frac{1}{R} \frac{\partial}{\partial r} \left( r^2 \frac{\partial R}{\partial r} \right) + \frac{1}{\Theta \sin \theta} \frac{\partial}{\partial \theta} \left( \sin \theta \frac{\partial \Theta}{\partial \theta} \right) + \frac{1}{\Phi \sin^2 \theta} \frac{\partial^2 \Phi}{\partial \phi^2} \right] + r^2 \sin^2 \theta \cdot \frac{2m_0}{\hbar^2} (E - V) = 0
\]

(2.68)

We may note that the second term in Equation (2.68) is a function of \( \theta \) only, whereas all the other terms are functions of either \( r \) or \( \phi \). We may then write that

\[
\frac{1}{\Phi} \frac{\partial^2 \Phi}{\partial \phi^2} = -m^2
\]

(2.69)

where \( m \) is a separation of variables constant. The solution to Equation (2.69) is of the form

\[
\Phi = e^{jm\phi}
\]

(2.70)

*The mass should be the rest mass of the two-particle system, but since the proton mass is much greater than the electron mass, the equivalent mass reduces to that of the electron.*

*Where \( m \) means the separation-of-variables constant developed historically. That meaning will be retained here even though there may be some confusion with the electron mass. In general, the mass parameter will be used in conjunction with a subscript.*