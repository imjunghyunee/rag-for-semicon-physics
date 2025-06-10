# 3.4 Density of States Function

The first factor, 2, takes into account the two spin states allowed for each quantum state; the next factor, \(\frac{1}{8}\), takes into account that we are considering only the quantum states for positive values of \(k_x\), \(k_y\), and \(k_z\). The factor \(4\pi k^2 dk\) is again the differential volume and the factor \((\pi a^3)\) is the volume of one quantum state. Equation (3.63) may be simplified to

\[
g_r(k)dk = \frac{\pi k^2 dk}{a^3}
\]

(3.64)

Equation (3.64) gives the density of quantum states as a function of momentum, through the parameter \(k\). We can now determine the density of quantum states as a function of energy \(E\). For a free electron, the parameters \(E\) and \(k\) are related by

\[
k^2 = \frac{2mE}{\hbar^2}
\]

(3.65a)

or

\[
k = \frac{1}{\hbar} \sqrt{2mE}
\]

(3.65b)

The differential \(dk\) is

\[
dk = \frac{1}{\hbar \sqrt{2E}} \sqrt{m} \, dE
\]

(3.66)

Then, substituting the expressions for \(k^2\) and \(dk\) into Equation (3.64), the number of energy states between \(E\) and \(E + dE\) is given by

\[
g_r(E)dE = \frac{\pi a^3}{\pi^3} \left(\frac{2mE}{\hbar^2}\right) \cdot \frac{1}{\hbar} \cdot \frac{\sqrt{m}}{\sqrt{2E}} \, dE
\]

(3.67)

Since \(\hbar = \frac{h}{2\pi}\), Equation (3.67) becomes

\[
g_r(E)dE = \frac{4\pi a^3}{h^3} \cdot (2m)^{3/2} \cdot \sqrt{E} \, dE
\]

(3.68)

Equation (3.68) gives the total number of quantum states between the energy \(E\) and \(E + dE\) in the crystal space volume of \(a^3\). If we divide by the volume \(a^3\), then we will obtain the density of quantum states per unit volume of the crystal. Equation (3.68) then becomes

\[
g(E) = \frac{4\pi(2m)^{3/2}}{h^3} \sqrt{E}
\]

(3.69)

The density of quantum states is a function of energy \(E\). As the energy of this free electron becomes small, the number of available quantum states decreases. This density function is really a double density, in that the units are given in terms of states per unit energy per unit volume.

----

**Objective:** Calculate the density of states per unit volume over a particular energy range.

**Example 3.3**

Consider the density of states for a free electron given by Equation (3.69). Calculate the density of states per unit volume with energies between 0 and 1 eV.