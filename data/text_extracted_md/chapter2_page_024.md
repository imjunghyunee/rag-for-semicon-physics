# Introduction to Quantum Mechanics

Since the wave function must be single-valued, we impose the condition that \( m \) is an integer, or

\[
m = 0, \pm 1, \pm 2, \pm 3, \ldots
\]

(2.71)

Incorporating the separation-of-variables constant, we can further separate the variables \(\theta\) and \(r\) and generate two additional separation-of-variables constants \(l\) and \(n\). The separation-of-variables constants \(n, l, m\) are known as quantum numbers. The parameter \(n\) is referred to as the principal quantum number, \(l\) is the azimuthal or angular quantum number, and \(m\) is the magnetic quantum number. The quantum numbers are related by

\[
n = 1, 2, 3, \ldots
\]

\[
l = n - 1, n - 2, n - 3, \ldots 0
\]

\[
|m| = l, (l - 1, \ldots, 0)
\]

(2.72)

Each set of quantum numbers corresponds to a quantum state that the electron may occupy. The electron energy may be written in the form

\[
E_n = -\frac{m_0 e^4}{(4\pi \varepsilon_0)^2 2\hbar^2 n^2}
\]

(2.73)

where again \(n\) is the principal quantum number. The negative energy indicates that the electron is bound to the nucleus and we again see that the energy of the bound electron is quantized. If the energy were to become positive, then the electron would no longer be a bound particle and the total energy would no longer be quantized. Since the parameter \(n\) in Equation (2.73) is an integer, the total energy of the electron can take on only discrete values. The quantized energy is again a result of the particle being bound in a finite region of space.

## Example 2.6

**Objective:** Calculate the first three energy levels of an electron for the one-electron atom.

### Solution

We have

\[
E_n = -\frac{m_0 e^4}{(4\pi \varepsilon_0)^2 2\hbar^2 n^2} = -\frac{(9.11 \times 10^{-31})(1.6 \times 10^{-19})^4}{[4\pi(8.85 \times 10^{-12})]^2(1.054 \times 10^{-34})^2 n^2}
\]

\[
= -21.726 \times 10^{-19} \frac{1}{n^2} \quad \text{or} \quad = -\frac{13.58}{n^2} \, \text{eV}
\]

For \(n = 1\): \(E_1 = -13.58 \, \text{eV}\)

For \(n = 2\): \(E_2 = -3.39 \, \text{eV}\)

For \(n = 3\): \(E_3 = -1.51 \, \text{eV}\)

### Comment

As the energy levels increase, the energy becomes less negative, which means that the electron is becoming less tightly bound to the atom.

### Exercise Problem

**Ex 2.6** In Example 2.6, assume the permittivity of free space, \(\varepsilon_0\), is replaced by the permittivity of a material where \(\varepsilon = \varepsilon_r \varepsilon_0\). Repeat the calculations in Example 2.6 if \(\varepsilon_r = 11.7\) (silicon).