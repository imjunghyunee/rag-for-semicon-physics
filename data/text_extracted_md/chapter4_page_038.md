# Exercise Problem

**Ex 4.12** Consider silicon at \( T = 300 \, \text{K} \) with doping concentrations of \( N_D = 8 \times 10^{15} \, \text{cm}^{-3} \) and \( N_A = 5 \times 10^{15} \, \text{cm}^{-3} \). Determine the position of the Fermi energy level with respect to \( E_c \). (\( kT = 0.0259 \, \text{eV} \))

----

We may develop a slightly different expression for the position of the Fermi level. We had from Equation (4.39) that \( n_0 = n_i \exp \left[ \frac{(E_F - E_i)}{kT} \right] \). We can solve for \( E_F - E_i \) as

\[
E_F - E_i = kT \ln \left( \frac{n_0}{n_i} \right)
\]

(4.65)

Equation (4.65) can be used specifically for an n-type semiconductor, where \( n_0 \) is given by Equation (4.60), to find the difference between the Fermi level and the intrinsic Fermi level as a function of the donor concentration. We may note that, if the net effective donor concentration is zero, that is, \( N_D - N_A = 0 \), then \( n_0 = n_i \) and \( E_F = E_i \). A completely compensated semiconductor has the characteristics of an intrinsic material in terms of carrier concentration and Fermi-level position.

We can derive the same types of equations for a p-type semiconductor. From Equation (4.19), we have \( p_0 = N_A \exp \left[ \frac{-(E_F - E_v)}{kT} \right] \), so that

\[
E_F - E_v = kT \ln \left( \frac{N_v}{p_0} \right)
\]

(4.66)

If we assume that \( N_A \gg n_i \), then Equation (4.66) can be written as

\[
E_F - E_v = kT \ln \left( \frac{N_v}{N_A} \right)
\]

(4.67)

The distance between the Fermi level and the top of the valence-band energy for a p-type semiconductor is a logarithmic function of the acceptor concentration: as the acceptor concentration increases, the Fermi level moves closer to the valence band. Equation (4.67) still assumes that the Boltzmann approximation is valid. Again, if we have a compensated p-type semiconductor, then the \( N_A \) term in Equation (4.67) is replaced by \( N_A - N_D \) or the net effective acceptor concentration.

We can also derive an expression for the relationship between the Fermi level and the intrinsic Fermi level in terms of the hole concentration. We have from Equation (4.40) that \( p_0 = n_i \exp \left[ \frac{-(E_F - E_i)}{kT} \right] \), which yields

\[
E_F - E_i = kT \ln \left( \frac{p_0}{n_i} \right)
\]

(4.68)

Equation (4.68) can be used to find the difference between the intrinsic Fermi level and the Fermi energy in terms of the acceptor concentration. The hole concentration \( p_0 \) in Equation (4.68) is given by Equation (4.62).