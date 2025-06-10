# Chapter 9: Metal–Semiconductor and Semiconductor Heterojunctions

**Comment**

In a heavily doped semiconductor, the depletion width is on the order of angstroms, so that tunneling is now a distinct possibility. For these types of barrier widths, tunneling may become the dominant current mechanism.

**Exercise Problem**

**Ex. 9.7** Calculate the space charge width of a rectifying metal–GaAs–semiconductor junction. Assume the n-type doping concentration is \( N_d = 7 \times 10^{18} \, \text{cm}^{-3} \) and the built-in potential barrier is \( V_{bi} = 0.80 \, \text{V} \).

The tunneling current has the form

\[
J_t \propto \exp\left(-\frac{e \phi_b}{E_{00}}\right)
\]

(9.29)

where

\[
E_{00} = \frac{e \hbar}{2} \sqrt{\frac{N_d}{\epsilon_s m_t^*}}
\]

(9.30)

The tunneling current increases exponentially with doping concentration.

## 9.2.3 Specific Contact Resistance

A figure of merit of ohmic contacts is the specific contact resistance, \( R_c \). This parameter is defined as the reciprocal of the derivative of current density with respect to voltage evaluated at zero bias. We may write

\[
R_c = \left( \frac{\partial J}{\partial V} \right)^{-1} \bigg|_{V=0} \, \Omega\text{-cm}^2
\]

(9.31)

We want \( R_c \) to be as small as possible for an ohmic contact.

For a rectifying contact with a low to moderate semiconductor doping concentration, the current–voltage relation is given by Equation (9.23) as

\[
J_n = A^* T^2 \exp\left(-\frac{e \phi_b}{kT}\right) \left[ \exp\left(\frac{eV}{kT}\right) - 1 \right]
\]

The thermionic emission current is dominant in this junction. The specific contact resistance for this case is then

\[
R_c = \left(\frac{kT}{e}\right) \exp\left(\frac{+e \phi_b}{kT}\right) \frac{1}{A^* T^2}
\]

(9.32)

The specific contact resistance decreases rapidly as the barrier height decreases.

For a metal–semiconductor junction with a high impurity doping concentration, the tunneling process will dominate. From Equations (9.29) and (9.30), the specific contact resistance is found to be

\[
R_c \propto \exp\left(\frac{+2 \sqrt{2 \epsilon_s m_t^*}}{\hbar} \cdot \frac{\phi_b}{\sqrt{N_d}}\right)
\]

(9.33)