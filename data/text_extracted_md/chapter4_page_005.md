# Applying the Boltzmann Approximation

The thermal-equilibrium density of electrons in the conduction band is found from:

\[
n_0 = \int_{E_c}^{\infty} \frac{4\pi(2m_n^*)^{3/2}}{h^3} \sqrt{E - E_c} \exp \left[ -\frac{(E - E_f)}{kT} \right] dE
\]

The integral of Equation (4.5) may be solved more easily by making a change of variable. If we let:

\[
\eta = \frac{E - E_c}{kT}
\]

then Equation (4.5) becomes:

\[
n_0 = \frac{4\pi(2m_n^*kT)^{3/2}}{h^3} \exp \left[ -\frac{(E_c - E_f)}{kT} \right] \int_0^{\infty} \eta^{1/2} \exp (-\eta) d\eta
\]

The integral is the gamma function, with a value of:

\[
\int_0^{\infty} \eta^{1/2} \exp (-\eta) d\eta = \frac{1}{2} \sqrt{\pi}
\]

Then Equation (4.7) becomes:

\[
n_0 = 2 \left( \frac{2\pi m_n^* kT}{h^2} \right)^{3/2} \exp \left[ -\frac{(E_c - E_f)}{kT} \right]
\]

We may define a parameter \( N_c \) as:

\[
N_c \equiv 2 \left( \frac{2\pi m_n^* kT}{h^2} \right)^{3/2}
\]

The parameter \( m_n^* \) is the density of states effective mass of the electron. The thermal-equilibrium electron concentration in the conduction band can be written as:

\[
n_0 = N_c \exp \left[ -\frac{(E_c - E_f)}{kT} \right]
\]

The parameter \( N_c \) is called the effective density of states function in the conduction band. If we were to assume that \( m_n^* = m_0 \), then the value of the effective density of states function at \( T = 300 \, \text{K} \) is \( N_c = 2.5 \times 10^{19} \, \text{cm}^{-3} \), which is of the order of magnitude of \( N_c \) for most semiconductors. If the effective mass of the electron is larger or smaller than \( m_0 \), then the value of the effective density of states function changes accordingly, but is still of the same order of magnitude.

## Example 4.1

**Objective:** Calculate the probability that a quantum state in the conduction band at \( E = E_c + kT/2 \) is occupied by an electron, and calculate the thermal-equilibrium electron concentration in silicon at \( T = 300 \, \text{K} \).

Assume the Fermi energy is 0.25 eV below the conduction band. The value of \( N_c \) for silicon at \( T = 300 \, \text{K} \) is \( N_c = 2.8 \times 10^{19} \, \text{cm}^{-3} \) (see Appendix B).

### Solution

The probability that a quantum state at \( E = E_c + kT/2 \) is occupied by an electron is given by:

\[
f(kT) = \frac{1}{1 + \exp \left[ \frac{E - E_f}{kT} \right]} \exp \left[ -\frac{(E - E_f)}{kT} \right] = \exp \left[ -\frac{(E_c + (kT/2) - E_f)}{kT} \right]
\]