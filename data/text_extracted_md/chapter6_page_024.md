## 6.3 Ambipolar Transport

The parameter \( \rho \) is the net charge density and the initial value is given by \( e\delta p \). We will assume that \( \delta p \) is uniform over a short distance at the surface. The parameter \( \epsilon \) is the permittivity of the semiconductor.

Taking the divergence of Ohm’s law and using Poisson’s equation, we find

\[
\nabla \cdot \mathbf{J} = \sigma \nabla \cdot \mathbf{E} = \frac{\sigma \rho}{\epsilon}
\tag{6.74}
\]

Substituting Equation (6.74) into the continuity equation, we have

\[
\frac{\sigma \rho}{\epsilon} = -\frac{\partial \rho}{\partial t} = -\frac{d \rho}{dt}
\tag{6.75}
\]

Since Equation (6.75) is a function of time only, we can write the equation as a total derivative. Equation (6.75) can be rearranged as

\[
\frac{d \rho}{dt} + \left( \frac{\sigma}{\epsilon} \right) \rho = 0
\tag{6.76}
\]

Equation (6.76) is a first-order differential equation whose solution is

\[
\rho(t) = \rho(0)e^{-t/\tau_d}
\tag{6.77}
\]

where

\[
\tau_d = \frac{\epsilon}{\sigma}
\tag{6.78}
\]

and is called the dielectric relaxation time constant.

----

### Objective: Calculate the dielectric relaxation time constant for a particular semiconductor.

**EXAMPLE 6.6**

Consider n-type silicon with a donor impurity concentration of \( N_d = 10^{16} \, \text{cm}^{-3} \).

#### Solution

The conductivity is found as

\[
\sigma \approx e \mu_n N_d = (1.6 \times 10^{-19})(1200)(10^{16}) = 1.92 \, (\Omega \cdot \text{cm})^{-1}
\]

where the value of mobility is the approximate value found from Figure 5.3. The permittivity of silicon is

\[
\epsilon = \epsilon_r \epsilon_0 = (11.7)(8.85 \times 10^{-14}) \, \text{F/cm}
\]

The dielectric relaxation time constant is then

\[
\tau_d = \frac{\epsilon}{\sigma} = \frac{(11.7)(8.85 \times 10^{-14})}{1.92} = 5.39 \times 10^{-13} \, \text{s}
\]

or

\[
\tau_d = 0.539 \, \text{ps}
\]