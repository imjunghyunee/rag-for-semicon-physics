## Objective: Calculate the probability that an energy state above \( E_F \) is occupied by an electron.

**Example 3.6**

Let \( T = 300 \, \text{K} \). Determine the probability that an energy level \( 3kT \) above the Fermi energy is occupied by an electron.

### Solution

From Equation (3.79), we can write

\[
f(E) = \frac{1}{1 + \exp\left(\frac{E - E_F}{kT}\right)} = \frac{1}{1 + \exp\left(\frac{3kT}{kT}\right)}
\]

which becomes

\[
f(E) = \frac{1}{1 + 20.09} = 0.0474 = 4.74\%
\]

### Comment

At energies above \( E_h \), the probability of a state being occupied by an electron can become significantly less than unity, or the ratio of electrons to available quantum states can be quite small.

### Exercise Problem

**Ex 3.6** Assume the Fermi energy level is 0.30 eV below the conduction band energy \( E_c \).

- Assume \( T = 300 \, \text{K} \). (a) Determine the probability of a state being occupied by an electron at \( E = E_F + kT/4 \). (b) Repeat part (a) for an energy state at \( E = E_F + kT \).

We can see from Figure 3.33 that the probability of an energy above \( E_F \) being occupied increases as the temperature increases and the probability of a state below \( E_F \) being empty increases as the temperature increases.

## Objective: Determine the temperature at which there is 1 percent probability that an energy state is empty.

**Example 3.7**

Assume that the Fermi energy level for a particular material is 6.25 eV and that the electrons in this material follow the Fermi–Dirac distribution function. Calculate the temperature at which there is a 1 percent probability that a state 0.30 eV below the Fermi energy level will not contain an electron.

### Solution

The probability that a state is empty is

\[
1 - f(E) = 1 - \frac{1}{1 + \exp\left(\frac{E - E_F}{kT}\right)}
\]