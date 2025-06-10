# CHAPTER 4 The Semiconductor in Equilibrium

where \( n_d \) is the density of electrons occupying the donor level and \( E_d \) is the energy of the donor level. The factor \(\frac{1}{2}\) in this equation is a direct result of the spin factor just mentioned. The \(\frac{1}{g}\) factor is sometimes written as \(1/g\), where \(g\) is called a degeneracy factor.

Equation (4.50) can also be written in the form

\[
n_d = N_d - N_d^+
\]

(4.51)

where \( N_d^+ \) is the concentration of ionized donors. In many applications, we will be interested more in the concentration of ionized donors than in the concentration of electrons remaining in the donor states.

If we do the same type of analysis for acceptor atoms, we obtain the expression

\[
p_a = \frac{N_a}{1 + \frac{1}{g} \exp \left( \frac{E_F - E_a}{kT} \right)} = N_a - N_a^-
\]

(4.52)

where \( N_a \) is the concentration of acceptor atoms, \( E_a \) is the acceptor energy level, \( p_a \) is the concentration of holes in the acceptor states, and \( N_a^- \) is the concentration of ionized acceptors. A hole in an acceptor state corresponds to an acceptor atom that is neutrally charged and still has an “empty” bonding position as we have discussed in Section 4.2.1. The parameter \( g \) is, again, a degeneracy factor. The ground state degeneracy factor \( g \) is normally taken as 4 for the acceptor level in silicon and gallium arsenide because of the detailed band structure.

## 4.4.2 Complete Ionization and Freeze-Out

The probability function for electrons in the donor energy state was just given by Equation (4.50). If we assume that \((E_d - E_F) \gg kT\) then

\[
n_d \approx \frac{N_d}{2} \exp \left( \frac{E_d - E_F}{kT} \right) = 2N_d \exp \left( \frac{-(E_d - E_F)}{kT} \right)
\]

(4.53)

If \((E_d - E_F) \gg kT\), then the Boltzmann approximation is also valid for the electrons in the conduction band so that, from Equation (4.11),

\[
n_0 = N_c \exp \left( \frac{-(E_c - E_F)}{kT} \right)
\]

We can determine the relative number of electrons in the donor state compared with the total number of electrons; therefore, we can consider the ratio of electrons in the donor state to the total number of electrons in the conduction band plus donor state. Using the expressions of Equations (4.53) and (4.11), we write

\[
\frac{n_d}{n_d + n_0} = \frac{2N_d \exp \left( \frac{-(E_d - E_F)}{kT} \right)}{2N_d \exp \left( \frac{-(E_d - E_F)}{kT} \right) + N_c \exp \left( \frac{-(E_c - E_F)}{kT} \right)}
\]

(4.54)