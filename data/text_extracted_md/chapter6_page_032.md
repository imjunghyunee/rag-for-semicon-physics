```markdown
proportional to the density of empty trap states. We can then write the electron capture rate as

\[
R_{cn} = C_n N_t [1 - f_t(E_t)] n
\]

(6.86)

where

- \( R_{cn} \) = capture rate (#/cm\(^3\)·s)
- \( C_n \) = constant proportional to electron-capture cross section
- \( N_t \) = total concentration of trapping centers
- \( n \) = electron concentration in the conduction band
- \( f_t(E_t) \) = Fermi function at the trap energy

The Fermi function at the trap energy is given by

\[
f_t(E) = \frac{1}{1 + \exp\left(\frac{E_t - E_f}{kT}\right)}
\]

(6.87)

which is the probability that a trap will contain an electron. The function \([1 - f_t(E_t)]\) is the probability that the trap is empty. In Equation (6.87), we have assumed that the degeneracy factor is 1, which is the usual approximation made in this analysis. However, if a degeneracy factor is included, it will eventually be absorbed in other constants later in the analysis.

For process 2, the rate at which electrons are emitted from filled traps back into the conduction band is proportional to the number of filled traps, so that

\[
R_{en} = E_n N_t f_t(E_t)
\]

(6.88)

where

- \( R_{en} \) = emission rate (#/cm\(^3\)·s)
- \( E_n \) = constant
- \( f_t(E_t) \) = probability that the trap is occupied

In thermal equilibrium, the rate of electron capture from the conduction band and the rate of electron emission back into the conduction band must be equal. Then

\[
R_{cn} = R_{en}
\]

(6.89)

so that

\[
E_n N_t f_t(E_t) = C_n N_t [1 - f_{t0}(E_t)] n_0
\]

(6.90)

where \( f_{t0} \) denotes the thermal-equilibrium Fermi function. Note that, in thermal equilibrium, the value of the electron concentration in the capture rate term is the equilibrium value \( n_0 \). Using the Boltzmann approximation for the Fermi function, we can find \( E_n \) in terms of \( C_n \) as

\[
E_n = n' C_n
\]

(6.91)
```