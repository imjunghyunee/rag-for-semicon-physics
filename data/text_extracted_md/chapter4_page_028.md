## 4.4 Statistics of Donors and Acceptors

The Fermi energy cancels out of this expression. Dividing by the numerator term, we obtain

\[
\frac{n_d}{n_u + n_0} = \frac{1}{1 + \frac{N_c}{2N_D} \exp \left( \frac{-(E_c - E_d)}{kT} \right)}
\]

(4.55)

The factor \((E_c - E_d)\) is just the ionization energy of the donor electrons.

----

### Objective: Determine the fraction of total electrons still in the donor states at \( T = 300 \, \text{K} \)

Consider phosphorus doping in silicon, for \( T = 300 \, \text{K} \), at a concentration of \( N_d = 10^{16} \, \text{cm}^{-3} \).

#### Solution

Using Equation (4.55), we find

\[
\frac{n_d}{n_0 + n_d} = \frac{1}{1 + \frac{2.8 \times 10^{19}}{2(10^{16})} \exp \left( \frac{-0.045}{0.0259} \right)} = 0.0041 = 0.41\%
\]

#### Comment

This example shows that there are very few electrons in the donor state compared with the conduction band. Essentially all of the electrons from the donor states are in the conduction band and, since only about 0.4 percent of the donor states contain electrons, the donor states are said to be completely ionized.

#### EXERCISE PROBLEM

**Ex 4.7** Repeat Example 4.7 for (a) \( T = 250 \, \text{K} \) and (b) \( T = 200 \, \text{K} \). (c) What can be said about the fraction as the temperature decreases?

----

At room temperature, then, the donor states are essentially completely ionized and, for a typical doping of \( 10^{16} \, \text{cm}^{-3} \), almost all donor impurity atoms have donated an electron to the conduction band.

At room temperature, there is also essentially complete ionization of the acceptor atoms. This means that each acceptor atom has accepted an electron from the valence band so that \( p_a \) is zero. At typical acceptor doping concentrations, a hole is created in the valence band for each acceptor atom. This ionization effect and the creation of electrons and holes in the conduction band and valence band, respectively, are shown in Figure 4.12.

The opposite of complete ionization occurs at \( T = 0 \, \text{K} \). At absolute zero degrees, all electrons are in their lowest possible energy state; that is, for an n-type semiconductor, each donor state must contain an electron, therefore \( n_d = N_d \) or \( N_d - n_d = 0 \). We must have, then, from Equation (4.50) that \(\exp \left[ \frac{(E_d - E_f)}{kT} \right] = 0\). Since \( T = 0 \, \text{K} \), this will occur for \(\exp(-\infty) = 0\), which means that \( E_f > E_d \). The Fermi energy level must be above the donor energy level at absolute zero. In the case of a p-type semiconductor at absolute zero temperature, the impurity atoms will not contain any electrons, so that the Fermi energy level must be below the acceptor energy state. The distribution of electrons among the various energy states, and hence the Fermi energy, is a function of temperature.