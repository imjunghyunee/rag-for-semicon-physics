# 4.6.1 Mathematical Derivation

The position of the Fermi energy level within the bandgap can be determined by using the equations already developed for the thermal-equilibrium electron and hole concentrations. **If we assume the Boltzmann approximation to be valid**, then from Equation (4.11) we have \( n_0 = N_c \exp \left[ \frac{-(E_c - E_F)}{kT} \right] \). We can solve for \( E_c - E_F \) from this equation and obtain

\[
E_c - E_F = kT \ln \left( \frac{N_c}{n_0} \right)
\]

(4.63)

where \( n_0 \) is given by Equation (4.60). If we consider an n-type semiconductor in which \( N_D \gg p_0 \), then \( n_0 \approx N_D \), so that

\[
E_c - E_F = kT \ln \left( \frac{N_c}{N_D} \right)
\]

(4.64)

The distance between the bottom of the conduction band and the Fermi energy is a logarithmic function of the donor concentration. As the donor concentration increases, the Fermi level moves closer to the conduction band. Conversely, if the Fermi level moves closer to the conduction band, then the electron concentration in the conduction band is increasing. We may note that if we have a compensated semiconductor, then the \( N_D \) term in Equation (4.64) is simply replaced by \( N_D - N_A \), or the net effective donor concentration.

----

## DESIGN EXAMPLE 4.12

**Objective:** Determine the required donor impurity concentration to obtain a specified Fermi energy.

Silicon at \( T = 300 \, \text{K} \) contains an acceptor impurity concentration of \( N_A = 10^{16} \, \text{cm}^{-3} \). Determine the concentration of donor impurity atoms that must be added so that the silicon is n-type and the Fermi energy is 0.20 eV below the conduction-band edge.

### Solution

From Equation (4.64), we have

\[
E_c - E_F = kT \ln \left( \frac{N_c}{N_D - N_A} \right)
\]

which can be rewritten as

\[
N_D - N_A = N_c \exp \left[ \frac{-(E_c - E_F)}{kT} \right]
\]

Then

\[
N_D - N_A = 2.8 \times 10^{19} \exp \left[ \frac{-0.20}{0.0259} \right] = 1.24 \times 10^{16} \, \text{cm}^{-3}
\]

or

\[
N_D = 1.24 \times 10^{16} + N_A = 2.24 \times 10^{16} \, \text{cm}^{-3}
\]

### Comment

A compensated semiconductor can be fabricated to provide a specific Fermi energy level.