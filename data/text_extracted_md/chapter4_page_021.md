# Example 4.5

**Objective:** Calculate the thermal equilibrium concentrations of electrons and holes for a given Fermi energy.

Consider silicon at \( T = 300 \, \text{K} \) so that \( N_c = 2.8 \times 10^{19} \, \text{cm}^{-3} \) and \( N_v = 1.04 \times 10^{19} \, \text{cm}^{-3} \). Assume that the Fermi energy is 0.25 eV below the conduction band. If we assume that the bandgap energy of silicon is 1.12 eV, then the Fermi energy will be 0.87 eV above the valence band.

## Solution

Using Equation (4.11), we have

\[
n_0 = (2.8 \times 10^{19}) \exp\left(\frac{-0.25}{0.0259}\right) = 1.8 \times 10^{15} \, \text{cm}^{-3}
\]

From Equation (4.19), we can write

\[
p_0 = (1.04 \times 10^{19}) \exp\left(\frac{-0.87}{0.0259}\right) = 2.7 \times 10^4 \, \text{cm}^{-3}
\]

## Comment

The change in the Fermi level is actually a function of the donor or acceptor impurity concentrations that are added to the semiconductor. However, this example shows that electron and hole concentrations change by orders of magnitude from the intrinsic carrier concentration as the Fermi energy changes by a few tenths of an electron-volt.

## Exercise Problem

**Ex 4.5** Determine the thermal-equilibrium concentrations of electrons and holes in silicon at \( T = 300 \, \text{K} \) if the Fermi energy level is 0.215 eV above the valence-band energy \( E_v \).

----

In the previous example, since \( n_0 > p_0 \), the semiconductor is n-type. In an n-type semiconductor, electrons are referred to as the majority carrier and holes as the minority carrier. By comparing the relative values of \( n_0 \) and \( p_0 \) in the example, it is easy to see how this designation came about. Similarly, in a p-type semiconductor where \( p_0 > n_0 \), holes are the majority carrier and electrons are the minority carrier.

We may derive another form of the equations for the thermal-equilibrium concentrations of electrons and holes. If we add and subtract an intrinsic Fermi energy in the exponent of Equation (4.11), we can write

\[
n_0 = N_c \exp\left(\frac{-(E_c - E_{Fi}) + (E_F - E_{Fi})}{kT}\right) \tag{4.38a}
\]

or

\[
n_0 = N_c \exp\left(\frac{-(E_c - E_{Fi})}{kT}\right) \exp\left(\frac{E_F - E_{Fi}}{kT}\right) \tag{4.38b}
\]

The intrinsic carrier concentration is given by Equation (4.20) as

\[
n_i = N_c \exp\left(\frac{-(E_c - E_{Fi})}{kT}\right)
\]