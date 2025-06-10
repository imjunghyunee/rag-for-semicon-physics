Then

\[
0.01 = 1 - \frac{1}{1 + \exp\left(\frac{5.95 - 6.25}{kT}\right)}
\]

Solving for \(kT\), we find \(kT = 0.06529 \, \text{eV}\), so that the temperature is \(T = 756 \, \text{K}\).

### Comment
The Fermi probability function is a strong function of temperature.

### EXERCISE PROBLEM
**Ex 3.7** Assume that \(E_F\) is 0.3 eV below \(E_c\). Determine the temperature at which the probability of an electron occupying an energy state at \(E = (E_c + 0.025) \, \text{eV}\) is \(8 \times 10^{-6}\).

----

We may note that the probability of a state a distance \(dE\) above \(E_F\) being occupied is the same as the probability of a state a distance \(dE\) below \(E_F\) being empty. The function \(f_F(E)\) is symmetrical with the function \(1 - f_F(E)\) about the Fermi energy, \(E_F\). This symmetry effect is shown in Figure 3.34 and will be used in the next chapter.

Consider the case when \(E - E_F \gg kT\) where the exponential term in the denominator of Equation (3.79) is much greater than unity. We may neglect the 1 in the denominator, so the Fermi–Dirac distribution function becomes

\[
f_F(E) \approx \exp\left(-\frac{E - E_F}{kT}\right)
\]

(3.80)

Equation (3.80) is known as the Maxwell–Boltzmann approximation, or simply the Boltzmann approximation, to the Fermi–Dirac distribution function. Figure 3.35 shows the Fermi–Dirac probability function and the Boltzmann approximation. This figure gives an indication of the range of energies over which the approximation is valid.

!Figure 3.34

**Figure 3.34** The probability of a state being occupied, \(f_F(E)\), and the probability of a state being empty, \(1 - f_F(E)\).